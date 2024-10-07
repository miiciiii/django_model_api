from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    arousal = serializers.CharField()
    dominance = serializers.CharField()
    continuous = serializers.DictField(child=serializers.FloatField())
