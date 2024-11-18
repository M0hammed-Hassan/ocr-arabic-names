import json
import torch
import torch.nn.functional as F
from rest_framework.views import APIView
from rest_framework import status
from .model import CRNNModel
from torchvision import transforms
from PIL import Image
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import HttpResponse
from django.shortcuts import render


class PredictionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def __init__(self, **kwargs) -> None:
        """
        Initialize the PredictionView class.

        This method loads the pre-trained CRNN model, transforms, and character mapping data.

        Parameters:
        - kwargs (dict): Additional keyword arguments passed to the parent class.

        Attributes:
        - model (CRNNModel): The pre-trained CRNN model for OCR.
        - transform (torchvision.transforms.Compose): A composition of image transformations.
        - idx_to_char (dict): A mapping from indices to characters for decoding the model's output.
        """
        super().__init__(**kwargs)
        self.model = self.load_model()
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((40, 96)),  
                transforms.ToTensor(),
            ]
        )
        self.idx_to_char = self.load_idx_to_char_data()

    def load_idx_to_char_data(self) -> dict:
        """
        Load the mapping from indices to characters for decoding the model's output.

        This method reads a JSON file containing the mapping from indices to characters and returns it as a dictionary.

        Parameters:
        None

        Returns:
        dict: A dictionary mapping indices to characters.
        """
        with open("idx_to_char.json", "r") as json_file:
            loaded_dict = json.load(json_file)
        return loaded_dict

    def load_model(self) -> torch.nn.Module:
        """
        Load the pre-trained CRNN model for OCR.

        This function initializes a CRNNModel instance with the specified number of classes,
        loads the model's weights from a checkpoint file, sets the model to evaluation mode,
        and returns the loaded model.

        Parameters:
        None

        Returns:
        CRNNModel: The loaded pre-trained CRNN model for OCR.
        """
        model = CRNNModel(num_classes=36)
        checkpoint = torch.load(
            "checkpoints/best.pth.tar", map_location=torch.device("cuda")
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def get(self, request, *args, **kwargs):
        """
        Handle GET requests to the PredictionView.

        This method renders an HTML template for uploading an image file.

        Parameters:
        - request (HttpRequest): The incoming HTTP request.
        - args (tuple): Additional positional arguments passed to the view.
        - kwargs (dict): Additional keyword arguments passed to the view.

        Returns:
        HttpResponse: An HTTP response containing the rendered HTML template for uploading an image file.
        """
        return render(request, "upload.html")

    def post(self, request, *args, **kwargs):
        """
        Handle POST requests to the PredictionView.

        This method processes an image file uploaded via a POST request, performs OCR using the loaded CRNN model,
        and returns the predicted name in reverse order.

        Parameters:
        - request (HttpRequest): The incoming HTTP request containing the uploaded image file.
        - args (tuple): Additional positional arguments passed to the view.
        - kwargs (dict): Additional keyword arguments passed to the view.

        Returns:
        HttpResponse: An HTTP response containing the predicted name in reverse order. If the image path is invalid,
        it returns a 400 Bad Request status with an error message.
        """
        try:
            image_path = request.POST.get("image_path")
            if image_path:
                image = Image.open(image_path)
                image = self.preprocess_image(image)
                with torch.no_grad():
                    output = self.model(image)
                    probs = F.softmax(output, dim=2)
                    _, predicted = torch.max(probs, 2)
                    predicted = predicted.permute(1, 0).contiguous().view(-1)
                    predicted_labels = [
                        idx.item() for idx in predicted if idx.item() != 35
                    ]
                    pred_name = "".join(
                        [self.idx_to_char[str(label)] for label in predicted_labels]
                    )
                return HttpResponse(f"Prediction: {pred_name[::-1]}")
        except Exception:
            return HttpResponse(
                "Invalid image path ", status=status.HTTP_400_BAD_REQUEST
            )

    def preprocess_image(self, image: Image.Image) -> torch.tensor:
        """
        Preprocess the input image for OCR using the loaded CRNN model.

        This function takes an input image, applies the necessary transformations,
        and prepares it for inference by the CRNN model.

        Parameters:
        - image (PIL.Image.Image): The input image to be processed.

        Returns:
        torch.Tensor: The preprocessed image tensor ready for inference. The tensor has a shape of (1, 1, 40, 96),
        where 1 represents the batch size, 1 represents the number of channels (grayscale), 40 represents the height,
        and 96 represents the width of the image.
        """
        image = self.transform(image).unsqueeze(0)
        return image
