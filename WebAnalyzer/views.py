# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.views.generic import TemplateView

from WebAnalyzer.models import ImageModel, ResultImage
from WebAnalyzer.serializers import ImageSerializer
from rest_framework.response import Response
from rest_framework import viewsets
from rest_framework import status

class ImageViewSet(viewsets.ModelViewSet):
    queryset = ImageModel.objects.all()
    serializer_class = ImageSerializer

    def get_queryset(self):
        queryset = self.queryset.order_by('-token')
        token = self.request.query_params.get('token', None)
        if token is not None:
            queryset = queryset.filter(token=token)
        return queryset

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        result_images = ResultImage.objects.filter(image_model=instance)
        result_image_urls = [request.build_absolute_uri(img.image.url) for img in result_images]

        return Response({
            'token': instance.token,
            'image1': request.build_absolute_uri(instance.image1.url),
            'image2': request.build_absolute_uri(instance.image2.url),
            'model_name': instance.model_name,
            'result_images': result_image_urls
        })
    def create(self, request, *args, **kwargs):
        image1 = request.FILES.get('image1')
        image2 = request.FILES.get('image2')
        model_name = request.data.get('model_name', '')

        if not image1 or not image2:
            return Response(
                {'error': 'Both image1 and image2 files are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        image_instance = ImageModel(image1=image1, image2=image2, model_name=model_name)
        image_instance.save()

        return Response({
            'message': 'Image uploaded and processing started!',
            'image_token': image_instance.token,
        }, status=status.HTTP_201_CREATED)

class ImageComparisonView(TemplateView):
    template_name = 'viewimages.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['images'] = ImageModel.objects.all()
        return context
