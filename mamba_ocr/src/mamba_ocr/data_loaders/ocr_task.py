import torch
class OcrTask:
    def get_batch(
        batch_size: int,
        pad_length: int,
        split: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve a batch of in-context OCR tasks.

        This function returns a tuple of 3 tensors: input, expected_output, and
        output_mask.

        The input tensor has the shape [B, L, D], where L equal to pad_length,
        B is batch_size, and D is the dimension of each token.
        Each token in the input tensor will either encode a pixel in an image,
        or a predicted letter.
        
        We can calculate D using the formula 
        
        D = 3 + D_
        
        

        Args:
            batch_size: An integer representing the number of samples to include
                in each batch.
            pad_length: _description_
            split: _description_

        Returns:
            A tuple ()
        """
        pass

    def get_batch_no_context(
        batch_size: int,
        pad_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves a batch of individual words.

        Args:
            batch_size (int): _description_
            pad_length (int): _description_

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
        """
        pass