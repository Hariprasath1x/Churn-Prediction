from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema, OpenApiResponse

from apps.predictions.serializers import PredictionInputSerializer
from ml.model_loader import model_manager
from ml.preprocessing import engineer_features
from ml.prediction import predict_catboost, predict_logistic_regression, predict_cox
from services.retention_engine import compute_retention_analysis

class PredictionView(APIView):
    @extend_schema(
        request=PredictionInputSerializer,
        responses={
            200: OpenApiResponse(description="Successful prediction and retention analysis"),
            400: OpenApiResponse(description="Validation Error"),
            500: OpenApiResponse(description="Model Loading or Inference Error")
        },
        summary="Generate churn prediction and retention analysis",
        description="Accepts raw customer features, engineers them, runs inference via CatBoost, LR, and Cox PH, and computes risk scoring and rule-based recommendations."
    )
    def post(self, request, *args, **kwargs):
        serializer = PredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        raw_data = serializer.validated_data
        
        # Load models
        try:
            cb_model = model_manager.get_catboost_model()
            lr_model = model_manager.get_logistic_model()
            cox_model = model_manager.get_cox_model()
        except RuntimeError as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        # Feature engineering
        try:
            df = engineer_features(raw_data)
        except Exception as e:
            return Response({"error": f"Feature engineering failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        # Inference
        cb_result = predict_catboost(cb_model, df)
        lr_result = predict_logistic_regression(lr_model, df)
        cox_result = predict_cox(cox_model, df)
        
        # Retention analysis
        try:
            analysis = compute_retention_analysis(
                raw=raw_data,
                cb_result=cb_result,
                cox_result=cox_result,
                lr_result=lr_result
            )
        except Exception as e:
            return Response({"error": f"Retention analysis failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        return Response(analysis, status=status.HTTP_200_OK)
