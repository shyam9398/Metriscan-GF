import os

# ---------------------------------------------------------
# PaddleOCR / PaddlePaddle memory configuration
# ---------------------------------------------------------

os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

from paddleocr import PaddleOCR


class OCRService:

    def __init__(self):
        print("Initializing PaddleOCR...")

        self.ocr = PaddleOCR(
            lang="en",
            device="cpu",
            enable_mkldnn=False,

            # -------------------------------------------------
            # MEMORY OPTIMIZATION
            #
            # Keep the core OCR pipeline:
            #   Text Detection
            #   Text Recognition
            #
            # Disable auxiliary models that are not required
            # for our packaged-product OCR workflow.
            # -------------------------------------------------

            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        print("PaddleOCR initialized successfully.")

    def extract_text(self, image_path: str):

        result = self.ocr.predict(
            image_path,

            # Explicitly keep the same configuration during
            # inference.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        ocr_results = []

        for res in result:

            data = res.json

            if isinstance(data, dict):
                data = data.get("res", data)

            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            boxes = data.get("rec_boxes", [])

            for i, text in enumerate(texts):

                confidence = None

                if i < len(scores):
                    confidence = float(scores[i])

                bbox = None

                if i < len(boxes):

                    current_box = boxes[i]

                    if hasattr(current_box, "tolist"):
                        bbox = current_box.tolist()
                    else:
                        bbox = current_box

                ocr_results.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox,
                    }
                )

        return ocr_results


# ---------------------------------------------------------
# SINGLE OCR INSTANCE
# ---------------------------------------------------------
#
# Keep exactly one OCR instance per backend process.
#
# Do NOT create PaddleOCR anywhere else.
# ---------------------------------------------------------

ocr_service = OCRService()