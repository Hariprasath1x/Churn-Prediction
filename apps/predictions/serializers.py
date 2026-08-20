from rest_framework import serializers

class PredictionInputSerializer(serializers.Serializer):
    gender = serializers.ChoiceField(choices=["Male", "Female"], default="Male")
    SeniorCitizen = serializers.IntegerField(min_value=0, max_value=1, default=0)
    Partner = serializers.ChoiceField(choices=["Yes", "No"], default="No")
    Dependents = serializers.ChoiceField(choices=["Yes", "No"], default="No")
    tenure = serializers.IntegerField(min_value=0, default=0)
    PhoneService = serializers.ChoiceField(choices=["Yes", "No"], default="Yes")
    MultipleLines = serializers.ChoiceField(choices=["Yes", "No", "No phone service"], default="No")
    InternetService = serializers.ChoiceField(choices=["DSL", "Fiber optic", "No"], default="DSL")
    OnlineSecurity = serializers.ChoiceField(choices=["Yes", "No", "No internet service"], default="No")
    OnlineBackup = serializers.ChoiceField(choices=["Yes", "No", "No internet service"], default="No")
    DeviceProtection = serializers.ChoiceField(choices=["Yes", "No", "No internet service"], default="No")
    TechSupport = serializers.ChoiceField(choices=["Yes", "No", "No internet service"], default="No")
    StreamingTV = serializers.ChoiceField(choices=["Yes", "No", "No internet service"], default="No")
    StreamingMovies = serializers.ChoiceField(choices=["Yes", "No", "No internet service"], default="No")
    Contract = serializers.ChoiceField(choices=["Month-to-month", "One year", "Two year"], default="Month-to-month")
    PaperlessBilling = serializers.ChoiceField(choices=["Yes", "No"], default="No")
    PaymentMethod = serializers.ChoiceField(
        choices=[
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],
        default="Electronic check"
    )
    MonthlyCharges = serializers.FloatField(min_value=0.0, default=0.0)
    TotalCharges = serializers.FloatField(min_value=0.0, default=0.0)
