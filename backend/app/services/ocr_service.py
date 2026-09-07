import os
import requests


class OCRService:
    """
    Remote PaddleOCR client.

    PaddleOCR runs in a separate Render service.
    """

    def __init__(self):

        self.ocr_url = os.getenv(
            "PADDLEOCR_SERVICE_URL"
        )

        if not self.ocr_url:
            raise RuntimeError(
                "PADDLEOCR_SERVICE_URL is not configured."
            )

        print("=" * 60)
        print("Remote PaddleOCR Service")
        print(f"URL: {self.ocr_url}")
        print("=" * 60)

    def extract_text(self, image_path: str):

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"OCR image does not exist: {image_path}"
            )

        print("Sending image to PaddleOCR service...")

        try:

            with open(image_path, "rb") as image_file:

                response = requests.post(
                    self.ocr_url,

                    files={
                        "file": (
                            os.path.basename(image_path),
                            image_file,
                            "image/jpeg",
                        )
                    },

                    timeout=180,
                )

            response.raise_for_status()

            data = response.json()

            if not data.get("success", False):

                raise RuntimeError(
                    "PaddleOCR service returned unsuccessful response."
                )

            ocr_results = data.get(
                "ocr_results",
                []
            )

            print(
                f"PaddleOCR returned "
                f"{len(ocr_results)} text blocks."
            )

            return ocr_results

        except requests.RequestException as exception:

            print(
                "PaddleOCR service error:",
                str(exception)
            )

            raise RuntimeError(
                f"PaddleOCR service unavailable: {exception}"
            )


# ============================================================
# SINGLETON
# ============================================================

ocr_service = OCRService()