import model_card_toolkit as mct
import uuid
from datetime import date
# import 06-micro_macro as quant_analysis


toolkit = mct.ModelCardToolkit()
model_card = toolkit.scaffold_assets()

model_card.model_details.name = 'Text Classification for legal documents'
model_card.model_details.overview = (
    'The primary focus of this model is to develop an automatic pipeline for the classification of the Gazzetta Ufficiale documents.')
model_card.model_details.owners = [
    mct.Owner(name= 'Model Cards Team', contact='model-cards@google.com')
]
model_card.model_details.references = [
    mct.Reference(reference='https://www.ital-ia2023.it/workshop/ai-per-la-pubblica-amministrazione')
]
model_card.model_details.version.name = str(uuid.uuid4())
model_card.model_details.version.date = str(date.today())

model_card.considerations.ethical_considerations = [mct.Risk(
    name=('Manual selection of the legal documents could create selection bias'),
    mitigation_strategy='Automate the selection process'
)]
model_card.considerations.limitations = [mct.Limitation(description='Legal documents classification')]
model_card.considerations.use_cases = [mct.UseCase(description='Legal documents classification')]
model_card.considerations.users = [mct.User(description=' professionals'), mct.User(description='ML researchers')]

model_card.model_parameters.data.append(mct.Dataset())
model_card.model_parameters.data[0].graphics.description = (
  f'{len(X_train)} rows with {len(X_train.columns)} features')


model_card.model_parameters.data.append(mct.Dataset())
model_card.model_parameters.data[1].graphics.description = (
  f'{len(X_test)} rows with {len(X_test.columns)} features')

model_card.quantitative_analysis.graphics.description = (
  'F1-Score and confusion matrix')

# add the f1-scoore value inside the mdoel card
model_card.quantitative_analysis.graphics.collection = [
    mct.Graphic(image=confusion_matrix),
    mct.Graphic(image=macro_f1_score)
]
toolkit.update_model_card(model_card)

# Return the model card document as an HTML page
html = toolkit.export_format()