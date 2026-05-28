class AgentvaultError(Exception):
    pass


class StoreError(AgentvaultError):
    pass


class FilterValidationError(AgentvaultError):
    pass


class EmbeddingError(AgentvaultError):
    pass
