"""AWS Lambda entry point for the API.

Mangum translates an API Gateway event into the ASGI call FastAPI expects, so
the same app runs under uvicorn locally and under Lambda when deployed.

CloudFront sends /api/* to API Gateway with the prefix intact (the site itself
owns /lakes/:id as a page route). api_gateway_base_path strips it, so FastAPI
sees /lakes/stats exactly as it does locally.
"""

from mangum import Mangum

from app.main import app

handler = Mangum(app, api_gateway_base_path="/api", lifespan="off")
