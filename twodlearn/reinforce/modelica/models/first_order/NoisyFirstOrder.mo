model NoisyFirstOrder
    // State start values
    parameter Real x_0 = 0;

    // parameters of the transfer function
    parameter Real K = 1.0;
    parameter Real tau = 1.0;
    
    // parameters of the noise distribution
    inner Modelica.Blocks.Noise.GlobalSeed globalSeed(enableNoise=true);
    parameter Real x_stddev = 0.3;
    parameter Real y_stddev = 0.3;

    // The states
    Real x(start = x_0, fixed=true);
    Modelica.Blocks.Noise.NormalNoise ex(mu=0.0, sigma=x_stddev, samplePeriod=0.1);
    Modelica.Blocks.Noise.NormalNoise ey(mu=0.0, sigma=y_stddev, samplePeriod=0.1);
    
    // The control signal
    input Real u;
    output Real y;

  equation
    der(x) = (K * u - x)/tau + ex.y;
    y = x + ey.y;
end NoisyFirstOrder;
