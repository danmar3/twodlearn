model FirstOrder
    // State start values
    parameter Real x_0 = 0;

    // parameters of the transfer function
    parameter Real K = 1.0;
    parameter Real tau = 1.0;

    // The states
    Real x(start = x_0, fixed=true);

    // The control signal
    input Real u;
    output Real y;

  equation
    der(x) = (K * u - x)/tau;
    y = x;
end FirstOrder;
