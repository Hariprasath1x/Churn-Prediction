from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema, OpenApiResponse, inline_serializer
from drf_spectacular.types import OpenApiTypes
from rest_framework.parsers import MultiPartParser
from rest_framework import serializers
from django.http import HttpResponse

import pandas as pd
import io

from ml.model_loader import model_manager
from ml.preprocessing import engineer_batch
from ml.prediction import predict_batch_catboost
from services.retention_engine import compute_batch_retention


class BatchPredictView(APIView):
    parser_classes = [MultiPartParser]

    @extend_schema(
        request=inline_serializer(
            name="BatchUploadSerializer",
            fields={
                "file": serializers.FileField(help_text="CSV file with customer data"),
            },
        ),
        responses={
            200: OpenApiTypes.BINARY,
            400: OpenApiResponse(description="Bad Request — missing file or invalid CSV"),
            500: OpenApiResponse(description="Server Error — model or inference failure"),
        },
        summary="Batch churn prediction via CSV upload",
        description=(
            "Upload a CSV with customer features. Returns an enriched CSV with "
            "churn probabilities, risk levels, and recommended actions."
        ),
    )
    def post(self, request, *args, **kwargs):
        file_obj = request.FILES.get("file")
        if not file_obj:
            return Response(
                {"error": "No file provided. Include a 'file' field with your CSV."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not file_obj.name.endswith(".csv"):
            return Response(
                {"error": "File must be a .csv file."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            df = pd.read_csv(file_obj)
        except Exception as e:
            return Response(
                {"error": f"Failed to parse CSV: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            cb_model = model_manager.get_catboost_model()

            # engineer_batch adds AvgMonthlySpend, NumServices,
            # HighValueCustomer, ContractRisk and returns FEATURE_COLUMNS
            df_engineered = engineer_batch(df)

            cb_probs = predict_batch_catboost(cb_model, df_engineered)

            # compute_batch_retention expects the engineered DF (which
            # contains Contract, HighValueCustomer, etc.) and probabilities
            df_enriched = compute_batch_retention(df_engineered, cb_probs)

            # Serialise to CSV and return as a downloadable attachment
            output = io.StringIO()
            df_enriched.to_csv(output, index=False)
            output.seek(0)

            response = HttpResponse(output.getvalue(), content_type="text/csv")
            response["Content-Disposition"] = (
                'attachment; filename="batch_predictions.csv"'
            )
            return response

        except Exception as e:
            return Response(
                {"error": f"Batch processing failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
