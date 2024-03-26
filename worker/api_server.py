import os
import time
import datetime
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from typing import List

from loguru import logger
from tools.environment import set_pod_status_env
from worker import graceful_exit,k8s_health
import threading


def api_middleware(app: FastAPI):
    rich_available = False
    try:
        if os.environ.get('WEBUI_RICH_EXCEPTIONS', None) is not None:
            import anyio  # importing just so it can be placed on silent list
            import starlette  # importing just so it can be placed on silent list
            from rich.console import Console
            console = Console()
            rich_available = True
    except Exception:
        pass

    @app.middleware("http")
    async def log_and_time(req: Request, call_next):
        ts = time.time()
        res: Response = await call_next(req)
        duration = str(round(time.time() - ts, 4))
        res.headers["X-Process-Time"] = duration
        endpoint = req.scope.get('path', 'err')
        logger.info('API {t} {code} {prot}/{ver} {method} {endpoint} {cli} {duration}'.format(
            t=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            code=res.status_code,
            ver=req.scope.get('http_version', '0.0'),
            cli=req.scope.get('client', ('0:0.0.0', 0))[0],
            prot=req.scope.get('scheme', 'err'),
            method=req.scope.get('method', 'err'),
            endpoint=endpoint,
            duration=duration,
        ))
        return res

    def handle_exception(request: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get('detail', ''),
            "body": vars(e).get('body', ''),
            "errors": str(e),
        }
        if not isinstance(e, HTTPException):  # do not print backtrace on known httpexceptions
            message = f"API error: {request.method}: {request.url} {err}"
        return JSONResponse(status_code=vars(e).get('status_code', 500), content=jsonable_encoder(err))

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)


class Api:
    def __init__(self, app: FastAPI):
        self.router = APIRouter()
        self.app = app
        api_middleware(self.app)
        self.add_api_route("/sysapi/v1/server-kill", self.kill, methods=["POST", "GET"])
        self.add_api_route("/sysapi/v1/server-restart", self.restart, methods=["POST", "GET"])
        self.add_api_route("/sysapi/v1/server-stop", self.stop, methods=["POST", "GET"])
        self.add_api_route("/sysapi/v1/server-health", self.health, methods=["GET"])

    def add_api_route(self, path: str, endpoint, **kwargs):
        return self.app.add_api_route(path, endpoint, **kwargs)

    def launch(self, server_name, port, root_path):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name, port=port, timeout_keep_alive=60, root_path=root_path)

    def kill(self):
        return Response("kill.")

    def restart(self):
        return Response("restart.")
    
    def health(self):
        
        return Response("health.")

    def stop(request):
        logger.info("stop webui func call")
        set_pod_status_env(graceful_exit.TERMINATING_STATUS)
        graceful_exit.wait_event()
        # # 获取当前进程的所有线程
        # threads = threading.enumerate()

        # # 打印当前进程的所有线程
        # print("Current threads in the process:")
        # for thread in threads:
        #     print(thread.name)
        th=threading.Thread(target=k8s_health._exit,name="exit thread")
        th.start()
        return Response("stop.")

def create_sys_api_server():
    app = FastAPI()
    api_server = Api(app)
    logger.info(f"system api server start···")
    api_server.launch(
        server_name="0.0.0.0",
        port= 7861,
        root_path=""
    )
