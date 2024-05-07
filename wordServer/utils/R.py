from django.http import JsonResponse


def ok(data, msg='操作成功'):
    return JsonResponse(data, msg)
