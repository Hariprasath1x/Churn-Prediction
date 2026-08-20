from django.db import models

class CustomerPrediction(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Raw inputs (subset for indexing/analysis)
    tenure = models.IntegerField()
    monthly_charges = models.FloatField()
    contract = models.CharField(max_length=50)
    
    # Predictions
    churn_probability = models.FloatField()
    churn_prediction = models.BooleanField()
    risk_level = models.CharField(max_length=20)
    priority_score = models.IntegerField()
    
    def __str__(self):
        return f"Prediction {self.id} - Risk: {self.risk_level}"

class AIStrategy(models.Model):
    prediction = models.ForeignKey(CustomerPrediction, on_delete=models.CASCADE, related_name='strategies')
    created_at = models.DateTimeField(auto_now_add=True)
    
    strategy_json = models.JSONField()
    
    def __str__(self):
        return f"Strategy for Prediction {self.prediction.id}"
