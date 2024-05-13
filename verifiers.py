from deepface import DeepFace

from datasets_adapters import KYCDatasetAdapter


class ModelVerifier:
    """ Class that implements functionality to verify model """
    model_name: str

    tp: int
    tn: int
    fp: int
    fn: int

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
    
    def _clear(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
    
    def _classify(self, _result: bool, _expected: bool):
        if _result:
            if _expected:
                self.tp += 1
            else:
                self.fp += 1
        else:
            if _expected:
                self.fn += 1
            else:
                self.tn += 1

    def execute(
            self,
            dataset_adapter: KYCDatasetAdapter,
    ):
        self._clear()

        total = 0
        print(f'Verification {self.model_name} on {str(dataset_adapter)}', end='\n\n')

        for verification_image_path in dataset_adapter.images_paths_iterator():
            print(f'Image path: {verification_image_path} ...'.ljust(50))
            for root_image_path in dataset_adapter.root_images_paths_iterator():
                print(f'Root image path: {root_image_path} ...'.ljust(50))
                result = DeepFace.verify(
                    root_image_path,
                    verification_image_path,
                    model_name=self.model_name,
                )['verified']
                expected = (verification_image_path.name in 
                            dataset_adapter.root_to_images_mapping[root_image_path.name])
                
                print(verification_image_path)
                print('    Result:'.ljust(50), result)
                print('    Expected:'.ljust(50), expected)

                self._classify(result, expected)

                total += 1
        
        self.counted_accuracy = (self.tp + self.tn) / (self.tp)
        print(f'\nTOTAL ACCURACY: {(self.tp + self.tn) / total}')
