import torch
class OcrTask:
    def get_d_color(self) -> int:
        pass

    def get_d_positional_encoding(self) -> int:
        pass

    def get_d_alphabet(self) -> int:
        pass

    def get_batch(
        self,
        batch_size: int,
        pad_length: int,
        split: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve a batch of in-context OCR tasks.

        This function returns a tuple of 3 tensors: input, expected_output, and
        output_mask.

        The input tensor has the shape [B, L, D_input], where L equal to
        pad_length, B is batch_size, and D is the dimension of each token.
        Each token in the input tensor will either encode a pixel in an image,
        or a predicted letter.
        The format of the input tensor samples will be as follows:

        `<image><prediction><image><prediction>`
        
        The output tensor has the shape [B, L, D_alph], and will contain samples
        formatted as follows:

        `<0-vectors><prediction><0-vectors><prediction>`
        
        The mask tensor has the shape [B, L], with each sample formatted as
        follows:

        `<0><1><0><1>`

        We can calculate D_input using the formula 
        
        D_input = D_color + D_pe + D_alph

        , where D_color is the number of color channels(1 for grayscale, 3 for
        full-color), D_pe is the dimension count of the positional encoding, and
        D_alph is the number of symbols in the alphabet.

        Note on padding: The padded-out inputs will all be 0. Since the mask is
        also 0 in the padded section, this should not affect training.

        Args:
            batch_size: The number of samples to include in each batch.
            pad_length: The length that all samples are padded to.
            split: A string equal to one of "train", "val", "test", which
                determines the source of the data.

        Returns:
            A tuple (input, expected_output, output_mask), where all 3 elements
            are tensors, and they have the shapes [B, L, D_input],
            [B, L, D_alph], and [B, L], respectively.
            The contents are described above.
        """
        pass

    def get_batch_no_context(
        self,
        batch_size: int,
        pad_length: int,
        split: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve a batch of out-of-context OCR tasks.

        This function returns a tuple of 3 tensors: input, expected_output, and
        output_mask.

        The input tensor has the shape [B, L, D_input], where L equal to
        pad_length, B is batch_size, and D is the dimension of each token.
        Each token in the input tensor will either encode a pixel in an image,
        or a predicted letter.
        The format of the input tensor samples will be as follows:

        `<image><prediction><padding>`
        
        The output tensor has the shape [B, L, D_alph], and will contain samples
        formatted as follows:

        `<0-vectors><prediction><0-vectors>`
        
        The mask tensor has the shape [B, L], with each sample formatted as
        follows:

        `<0><1><0>`

        We can calculate D_input using the formula 
        
        D_input = D_color + D_pe + D_alph

        , where D_color is the number of color channels(1 for grayscale, 3 for
        full-color), D_pe is the dimension count of the positional encoding, and
        D_alph is the number of symbols in the alphabet.

        Note on padding: The padded-out inputs will all be 0. Since the mask is
        also 0 in the padded section, this should not affect training.

        Args:
            batch_size: The number of samples to include in each batch.
            pad_length: The length that all samples are padded to.
            split: A string equal to one of "train", "val", "test", which
                determines the source of the data.

        Returns:
            A tuple (input, expected_output, output_mask), where all 3 elements
            are tensors, and they have the shapes [B, L, D_input],
            [B, L, D_alph], and [B, L], respectively.
            The contents are described above.
        """
        pass
