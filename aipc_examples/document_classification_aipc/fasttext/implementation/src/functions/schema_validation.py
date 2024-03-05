
# implementation of the frictionless validation logic
import datajudge as dj
class ValidationSchema:
    """
    This is a class that represents a validation schema. 
    It takes in a configuration object and initializes a LoggingUtils object and a metadata store object. 
    It also calls the find_resources method with the schema_file and domain_data_path parameters.
    """
    def __init__(self, configs):
        self.config = configs
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

def validate_resource():
    #TODO load validation configurations
    validation_config = {}
    validator = ValidationSchema(validation_config)
    validator.validate_resource()