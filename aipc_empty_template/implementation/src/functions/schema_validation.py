
# implementation of the frictionless validation logic

class ValidationSchema:
    """
    This is a class that represents a validation schema. 
    It takes in a configuration object and initializes a LoggingUtils object and a metadata store object. 
    It also calls the find_resources method with the schema_file and domain_data_path parameters.
    """
    def __init__(self, configs):
        self.config = configs
        self.logging = LoggingUtils(configs)
        self.metadata_store = dj.StoreConfig(
            title="Local Metadata Store",
            name="local_md",
            uri=self.config.validation_schema_source["data_judge_run_folder"],
            type="local",
        )
        self.store_local = dj.StoreConfig(
            name="local",
            uri=self.config.validation_schema_source["data_judge_run_folder"],
            type="local",
            isDefault=True,
        )
        self.RUN_CFG = dj.RunConfig(
            inference=[{"library": "frictionless"}],
            validation=[
                {
                    "library": "frictionless",
                    "execArgs": {
                        "limit_errors": int(
                            self.config.validation_schema_source[
                                "threshold_errors_to_check"
                            ]
                        )
                    },
                },
                {"library": "duckdb"},
            ],
            profiling=[
                {
                    "library": "pandas_profiling",
                    "execArgs": {"minimal": True, "progress_bar": False},
                }
            ],
        )

    def obtain_full_constraints(self, fields, resource_name):
        """
        This function returns a list of constraints for a given resource. The constraints are of type `dj.ConstraintFullFrictionless` and are defined by the input parameters `fields` and `resource_name`.
        @param self - the class instance
        @param fields - the fields to be used in the constraints
        @param resource_name - the name of the resource to apply the constraints to
        @return a list of constraints
        """        
        return [
            dj.ConstraintFullFrictionless(
                name="full_constraints",
                title="full_constraints",
                resources=[resource_name],
                weight=1,
                type="frictionless_schema",
                tableSchema=fields,
            )
        ]

    def validation_schema(self, full_constraints, resource_local):
        """
        This method validates a data source based on its schema. 
        It creates a run using the provided metadata store and local store. 
        It then validates the run using the provided constraints and logs the report. 
        If an exception is thrown, it logs the error and prints the exception message.
        @param self - the object instance
        @param full_constraints - the constraints to validate the data source against
        @param resource_local - the local resource to validate
        @return None
        """
        try:
            client = dj.Client(
                metadata_store=self.metadata_store, store=[self.store_local]
            )
            run = client.create_run(
                [resource_local], self.RUN_CFG, experiment="EXP-NAME"
            )
            with run:
                run.validate(constraints=full_constraints, parallel=True)
                run.log_report()
                run.persist_report()
        except Exception as e:
            self.logging.log_generator_error(e, False)
            print(e)

    def generate_constraints_resources(self, data_path, schema_path, resource_name):
        """
        Given a data path, schema path, and resource name, generate constraints and resources.
        @param self - the class instance
        @param data_path - the path to the data
        @param schema_path - the path to the schema
        @param resource_name - the name of the resource
        @return A tuple containing the full constraints and the local resource.
        """
        fields = read_schema(schema_path, resource_name)
        if len(fields) == 0:
            msg = "There is no schema defined for {} ".format(resource_name)
            raise Exception(msg)
        full_constraints = self.obtain_full_constraints(fields, resource_name)

        resource_local = dj.DataResource(
            path=os.path.relpath(data_path),
            name=resource_name,
            store="local",
            schema=fields,
        )
        return full_constraints, resource_local

    def find_resources(self, schema_file, domain_data_path):
        """
        Given a schema file and a domain data path, find all resources in the domain data path and validate them against the schema file.
        @param self - the class instance
        @param schema_file - the schema file to validate against
        @param domain_data_path - the path to the domain data
        @return None
        """
        data_sources = os.listdir(domain_data_path)
        for file in data_sources:
            data_path = os.path.join(domain_data_path, file)
            if (
                os.path.isfile(data_path)
                and file != self.config.parser_source["original_schema_file"]
            ):
                self.validation_schema_wrapper(file.lower, schema_file, data_path)

    def validate_resource(self, resource=None, type="source"):
        """
        This method validates a resource, given a resource and a type. 
        If the type is "source", it will use the package source and data source from the configuration file to validate the resource. 
        If the type is "target", it will use the package target and data target from the configuration file to validate the resource. 
        It will then check if the resource exists and if it does, it will call the validation_schema_wrapper method with the resource, schema file, and data path. 
        If the resource is None, it will call the find_resources method with the schema file and domain data path.
        @param self - the object instance
        @param resource - the resource to validate
        @param type - the type of resource to validate (source or target)
        @return None
        """
        if type == "source":
            packagename = self.config.domain["package_source"]
            domain_data_path = self.config.validation_schema_source["data_source"]
        else:
            packagename = self.config.domain["package_target"]
            domain_data_path = self.config.validation_schema_target["data_target"]
        schema_file = self.config.domain["domain_settings_path"] + packagename + ".json"
        if resource is not None:
            data_path = os.path.join(domain_data_path, resource)
            if os.path.isfile(data_path):
                self.validation_schema_wrapper(resource, schema_file, data_path)
        else:
            self.find_resources(schema_file, domain_data_path)

    def validation_schema_wrapper(self, resource, schema_file, data_path):
        """
        This function is a wrapper for a validation schema. 
        It generates constraints and resources based on a given schema file and data path, and then validates the constraints against the resources.
        @param self - the object instance
        @param resource - the resource to validate
        @param schema_file - the schema file to use for validation
        @param data_path - the path to the data to validate
        @return None
        """
        full_constraints, resource_local = self.generate_constraints_resources(
            data_path, schema_file, resource.lower()
        )
        self.logging.log_generator(
            "Starting to validate {} {}...".format(resource.lower(), data_path), False
        )
        self.validation_schema(full_constraints, resource_local)
        self.logging.log_generator("Ending validation {}...".format(resource.lower()))

        
def validate_resource():
    #TODO load validation configurations
    validation_config = {}
    validator = ValidationSchema(validation_config)
    validator.validate_resource()