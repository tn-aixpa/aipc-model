import os
import boto3
import json
import logging
import csv
import re
from datetime import datetime
import mlrun
from mlrun.execution import MLClientCtx
import pandas as pd

# 01-parse_ipzs.py refactored into callable functions
# Input files are assumed to be stored in S3, CSV file is read with pandas

def parse_ipzs(context: MLClientCtx, bucket_name: str, idPrefix = "ipzs-", limit = 10, max_documents = None):
    """
    Process IPZS CSV data and generate a JSON file with labels and IDs. Accepts:
    - bucket_name: S3 bucket containing atti_SG_materie.csv and the IPSZ documents index
    - idPrefix: defaults to ipzs-
    - limit: defaults to 10
    - max_documents: maximum number of documents to process (for testing)

    The resulting JSON file has the following structure:

    ```
    {
        "20000111_000G0003": {
            "text": "Interventi straordinari nel settore dei beni e delle attivita' culturali. ",
            "id": "ipzs-ml-20000111_000G0003",
            "labels": [
                "A0400",
                "C0I90"
            ]
        },
    ...
    }
    ```
    """
    logging.basicConfig(level=logging.INFO)
    csvfilename = "atti_SG_materie.csv" #NOTE: skipped prepending bucket name

    codePatt = re.compile("([^_]+_[^_]+)_([^_]+)_([^_]+)\.json")

    dayIndex = {}
    index = {}

    s3_endpoint_url = "https://minio-api.digitalhub-test.smartcommunitylab.it/" #os.environ.get("S3_ENDPOINT_URL")
    logging.info("s3_endpoint_url: " + s3_endpoint_url)
    logging.info("s3_endpoint_url: " + " provaaa")

    s3 = boto3.resource('s3',
                        endpoint_url=s3_endpoint_url,
                        aws_access_key_id='minio',
                        aws_secret_access_key='digitalhub-test',
                        aws_session_token=None,
                        config=boto3.session.Config(signature_version='s3v4'))

    ipzs_bucket = s3.Bucket(bucket_name)

    iterations = 0 #NOTE: used to limit the number of documents to be processed
    for obj in ipzs_bucket.objects.all():
        if obj.key.endswith("json"):
            fileNamePath = obj.key #NOTE: skipped prepending bucket name
            item = obj.key.split("/")[-1]
            m = codePatt.match(item)
            if m:
                if m.group(1) in index:
                    logging.info(m.group(1) + " already exists")
                    logging.info(fileNamePath + " --- " + index[m.group(2)])
                index[m.group(1)] = fileNamePath
                iterations = iterations + 1
            else:
                logging.error("ERR: " + item)
        if max_documents and iterations == max_documents:
            break

    allObjs = []
    count = {}
    logging.info("here1" + f"s3://{bucket_name}/{csvfilename}") 
    df = pd.read_csv(
        f"s3://{bucket_name}/{csvfilename}",
        delimiter=",",
        quotechar='"',
        storage_options={"client_kwargs": {"endpoint_url": s3_endpoint_url}, 'key':'minio', 'secret':'digitalhub-test'}
    )

    for row in df.itertuples(index=False):
        day = row[1]
        code = row[0]
        category = row[3]
        d = datetime.strptime(day.lower(), "%d-%b-%y").date()
        code = datetime.strftime(d, "%Y%m%d") + "_" + code
        if not code in index:
            logging.error("%s not in index" % code)
            continue

        s3_object = s3.Object(bucket_name, index[code])
        data = json.loads(s3_object.get()["Body"].read().decode("utf-8"))

        titolo = data['metadati']['titoloDoc']
        titolo = re.sub(r"\n", " ", titolo)
        titolo = re.sub(r" +", " ", titolo)
        thisObj = {}
        thisObj['text'] = titolo
        thisObj['id'] = idPrefix + code
        thisObj['labels'] = [category]
        if category not in count:
            count[category] = 0
        count[category] += 1
        allObjs.append(thisObj)

    logging.info(f"Length of result: {len(allObjs)}")
    if limit > 1:
        for index in count:
            if count[index] < limit:
                allObjs = [x for x in allObjs if index not in x['labels']]

    logging.info(f"Length of result: {len(allObjs)}")
    jsonString = json.dumps(allObjs, indent=4)
    context.log_artifact("preprocessed_data", body=jsonString, local_path="preprocessed_data.json", upload=True, format="json")
