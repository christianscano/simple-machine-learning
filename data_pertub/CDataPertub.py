from abc import ABC, abstractmethod


class CDataPertub(ABC):

    @abstractmethod
    def data_perturbation(self, x):
        """
        It takes a flat vector x and return a perturbed version
        """
        raise NotImplementedError


    def pertub_dataset(self, X):
        Xp = X.copy()
        
        for index in range(Xp.shape[1]):
            Xp[index, :] = self.data_perturbation(Xp[index, :])

        return Xp
    


    
            

    