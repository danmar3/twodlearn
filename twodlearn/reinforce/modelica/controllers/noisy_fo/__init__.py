import twodlearn.reinforce.modelica.models.first_order as first_order


class NoisyFO(first_order.NoisyFirstOrderModel):
    def set_parameters(self, nominal_x=None, k=None, tau=None,
                       x_stddev=None, y_stddev=None):
        self.parameters.model.set([k, tau])
        self.parameters.dist.set([x_stddev, y_stddev])
        self._nominal_x = nominal_x
        if nominal_x is None:
            self.reset()
        else:
            self.reset(x0=nominal_x)

    def __init__(self, dt):
        self._nominal_x = None
        super(NoisyFO, self).__init__(dt)

    def __call__(self, x=None, t=None):
        if self._nominal_x is None:
            self.step()
        else:
            u = self._nominal_x/self.parameters.model.data[0]
            self.step(u)
        return self.x.data
