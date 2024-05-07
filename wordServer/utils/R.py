from django.http import JsonResponse


def ok(data, msg='操作成功'):
    return JsonResponse({
        'code': 200,
        'data': data,
        'msg': msg,
    })
