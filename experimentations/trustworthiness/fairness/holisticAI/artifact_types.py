import yaml

class Data():
    """
    The primary artifact fed into the training algorithm to fit the best model
    """
    def __init__(self, filepath):
        self.filepath = filepath
    
    
class Report():
    """
    Structured information about the results obtained after applying a function
    """
    def __init__(self, filepath):
        self.filepath = filepath
        
    
    
class Model():
    """
    Either a single file or multiple files constituting the model
    """
    def __init__(self, model_path: str):
        self.model_path =  model_path
        
    
class Configuration():
    """
    Declarative specifications used to orchestrate the execution of individual components or pipelines
    """
    def __init__(self, filepath):
        self.filepath = filepath
    
    def load_config(self):
        with open(self.filepath, "r") as file:
            config = yaml.safe_load(file) 
        return config
    
    def save_config(self, configs):
        with open(self.filepath, "w") as file:
            yaml.dump(configs, file, default_flow_style=False)
            
    
class Status():
    """
    A blocking artifact that defines the pipeline execution
    """
    def __init__(self, status):
        self.current_status = status
    
    def change_status(self, new_status):
        self.current_status = new_status
            
class Documentation():
    """
    Files that contain human-readable content
    """
    def __init__(self):
        pass
    
class Function():
    """
    An implementation function
    """
    def __init__(self):
        pass

class Service():
    """
    Service encapsulating the model and parameters for serving it, matching evaluation parameters
    """
    def __init__(self):
        pass
        
class Logs():
    """
    Generated during each phase of the AI lifecycle. Logging of events or Observability services
    """
    def __init__(self):
        pass