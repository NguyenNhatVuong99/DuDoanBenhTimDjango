from django.http import JsonResponse
from django.shortcuts import render
from rest_framework import views, status
from .helpers import custom_response, parse_request, predict_target, trainSet, k


class HeartAPIView(views.APIView):
    def post(self, request):
        data = parse_request(request)
        keys = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

        point = [data.get(key, 0) for key in keys]
        test_data = [point]
        predicted_targets = predict_target(test_data, trainSet, k)

        result = predicted_targets[0]
        return custom_response('Successfully!', 'Success', result, 201)


def home(request):
    return render(request, 'home.html')
