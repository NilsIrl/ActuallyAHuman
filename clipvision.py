import os
import cv2
import base64
import io
import json
from PIL import Image
from vespa.deployment import VespaCloud
from vespa.application import Vespa
from vespa.package import ApplicationPackage

def encode_image_to_base64(image):
    """
    Convert a BGR image (numpy array) to a base64-encoded JPEG.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

# --------------------------------------------------
# Configuration: Replace these values with your own.
# --------------------------------------------------
tenant_name = "omnom"             # e.g., "vespa-team"
application = "hybridsearch"             # your Vespa application name
api_key = os.getenv("MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE7BnL1kEWylG+7NNcGMSdPLdwJ++Td6qpOuhB5vL4ObuljzjZAv+pj6Rtlu3M1IJDbXSFrUiY6P9uDe4rbHE2BQ==")  # ensure your API key is set in the environment

if api_key is not None:
    # Make sure newline characters are correctly interpreted.
    api_key = api_key.replace(r"\n", "\n")

# Create a minimal Vespa application package.
package = ApplicationPackage(name=application)
# (In practice, your package should include a schema with fields for floorplan and navigation.)

# --------------------------------------------------
# Deploy Application to Vespa Cloud
# --------------------------------------------------
vespa_cloud = VespaCloud(
    tenant=tenant_name,
    application=application,
    key_content=api_key,
    application_package=package,
)

print("Deploying application to Vespa Cloud...")
app_instance = vespa_cloud.deploy()  # Blocks until deployment is complete.
endpoint = app_instance.get_mtls_endpoint()
print("Deployment complete. Endpoint:", endpoint)

# --------------------------------------------------
# Load Floorplan Image
# --------------------------------------------------
floorplan_image = cv2.imread("floorplan.jpg")
if floorplan_image is None:
    raise ValueError("Could not load floorplan.jpg. Please check the file path.")

# --------------------------------------------------
# Generate or Provide Floorplan Description and Navigation Instructions
# --------------------------------------------------
# In a real system, you might use a vision model (e.g. BLIP-2 or GPT-4 Vision mini)
# to generate a caption from the floorplan image. For this example, we define them:
floorplan_caption = (
    "The floorplan shows a rectangular layout with a main corridor on the left, "
    "an elevator bank near the center, and conference rooms on the right."
)
navigation_instructions = (
    "From the main entrance, walk straight for 15 meters, then turn right at the corridor; "
    "follow the corridor for 10 meters and enter Conference Room A on your left."
)

# --------------------------------------------------
# Feed Document to Vespa
# --------------------------------------------------
doc = {
    "id": "floorplan-1",
    "fields": {
        "floorplan": floorplan_caption,
        "navigation": navigation_instructions
        # Optionally, you could also store the encoded image:
        # "image": encode_image_to_base64(floorplan_image)
    }
}

print("Feeding floorplan document to Vespa...")
app_instance.feed_document("doc", doc)
print("Document fed.")

# --------------------------------------------------
# Query Vespa for Navigation Instructions
# --------------------------------------------------
# In this example, we simply query using a plain text query.
query = "navigate from main entrance to Conference Room A"
response = app_instance.query(
    yql="select * from sources * where userQuery() limit 1",
    query=query,
    ranking="bm25"  # Adjust rank profile as needed.
)

print("Vespa Query Response:")
print(json.dumps(response.get_json(), indent=2))