## this class provides a common class to which all MARL environment should adhere
## The purpose is to train all the environment using the same code


class CommonInterface(object):
    
    def step(self,actions):
        raise NotImplementedError
    

    def reset(self):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError
    def sample():
        raise NotImplementedError