from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    status = 400
    code = "bad_request"

    def __init__(self, message: str, **details: object) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def as_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=self.status,
            content={
                "error": {
                    "code": self.code,
                    "message": self.message,
                    "details": self.details,
                }
            },
        )


class Forbidden(AppError):
    status = 403
    code = "forbidden"


class NotFound(AppError):
    status = 404
    code = "not_found"


class Conflict(AppError):
    status = 409
    code = "conflict"


class Invalid(AppError):
    status = 422
    code = "invalid"


class Unavailable(AppError):
    status = 503
    code = "unavailable"


class ErrorHandlers:
    @staticmethod
    def install(api: FastAPI) -> None:
        @api.exception_handler(AppError)
        async def _handle(_: Request, exc: AppError) -> JSONResponse:
            return exc.as_response()
