

class ContextMixin(object):

    _context_features = None
    _context_model = None

    @property
    def context_features(self):
        return self._context_features

    @property
    def context_model(self):
        return self._context_model

    def context_encode(self, inputs, training=False):
        return self.context_model.__call__(
                inputs[self.context_features],
                training=training
        )
