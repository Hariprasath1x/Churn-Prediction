from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema, OpenApiResponse
from django.http import HttpResponse

from apps.predictions.serializers import PredictionInputSerializer
from ml.model_loader import model_manager
from ml.preprocessing import engineer_features
from ml.prediction import predict_catboost, predict_logistic_regression, predict_cox
from services.retention_engine import compute_retention_analysis
from services.ai_retention_advisor import generate_ai_strategy
from services.report_generator import generate_report

class AIStrategyView(APIView):
    @extend_schema(
        request=PredictionInputSerializer,
        responses={
            200: OpenApiResponse(description="Successful AI strategy generation"),
            400: OpenApiResponse(description="Validation Error"),
            500: OpenApiResponse(description="Server Error")
        },
        summary="Generate AI-powered retention strategy",
        description="Re-runs inference and generates a personalized retention strategy using Groq LLM."
    )
    def post(self, request, *args, **kwargs):
        serializer = PredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        raw_data = serializer.validated_data
        
        try:
            cb_model = model_manager.get_catboost_model()
            lr_model = model_manager.get_logistic_model()
            cox_model = model_manager.get_cox_model()
            
            df = engineer_features(raw_data)
            
            cb_result = predict_catboost(cb_model, df)
            lr_result = predict_logistic_regression(lr_model, df)
            cox_result = predict_cox(cox_model, df)
            
            analysis = compute_retention_analysis(
                raw=raw_data,
                cb_result=cb_result,
                cox_result=cox_result,
                lr_result=lr_result
            )
        except Exception as e:
            return Response({"error": f"Failed to compute baseline analysis: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        strategy = generate_ai_strategy(analysis=analysis, raw=raw_data)
        
        if strategy.get("error") and not strategy.get("fallback_used"):
            return Response(strategy, status=status.HTTP_503_SERVICE_UNAVAILABLE)
            
        return Response(strategy, status=status.HTTP_200_OK)


class ReportView(APIView):
    @extend_schema(
        request=PredictionInputSerializer,
        responses={
            200: OpenApiResponse(description="Plain text analysis report"),
            400: OpenApiResponse(description="Validation Error"),
            500: OpenApiResponse(description="Server Error")
        },
        summary="Generate plain-text retention report",
        description="Generates a downloadable text report for a customer's retention analysis. Also computes AI strategy internally if possible."
    )
    def post(self, request, *args, **kwargs):
        serializer = PredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
        raw_data = serializer.validated_data
        
        try:
            cb_model = model_manager.get_catboost_model()
            lr_model = model_manager.get_logistic_model()
            cox_model = model_manager.get_cox_model()
            
            df = engineer_features(raw_data)
            
            cb_result = predict_catboost(cb_model, df)
            lr_result = predict_logistic_regression(lr_model, df)
            cox_result = predict_cox(cox_model, df)
            
            analysis = compute_retention_analysis(
                raw=raw_data,
                cb_result=cb_result,
                cox_result=cox_result,
                lr_result=lr_result
            )
        except Exception as e:
            return Response({"error": f"Failed to compute baseline analysis: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        # Optional AI strategy
        strategy = generate_ai_strategy(analysis=analysis, raw=raw_data)
        
        report_text = generate_report(raw=raw_data, analysis=analysis, ai_strategy=strategy)
        
        response = HttpResponse(report_text, content_type='text/plain')
        response['Content-Disposition'] = 'attachment; filename="retention_report.txt"'
        return response
