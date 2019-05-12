model SaturatedNoisyFO
    // State start values
    parameter Real x_0 = 0;

    // parameters of the transfer function
    parameter Real K = 1.0;
    parameter Real tau = 1.0;
    
    // parameters of the noise distribution
    inner Modelica.Blocks.Noise.GlobalSeed globalSeed(enableNoise=true);
    parameter Real x_stddev = 0.3;
    parameter Real y_stddev = 0.3;

    // parameters for the saturation
    parameter Real y_min = -1.0;
    parameter Real y_max = 1.0;
    // Real delta_y(start = y_max - y_min, fixed=true);
    // Real mean_y(start = (y_max + y_min)/2.0, fixed=true);
    
    // The states
    Real x(start = x_0, fixed=true);
    Modelica.Blocks.Noise.NormalNoise ex(mu=0.0, sigma=x_stddev, samplePeriod=0.1);
    Modelica.Blocks.Noise.NormalNoise ey(mu=0.0, sigma=y_stddev, samplePeriod=0.1);
    
    // The control signal
    input Real u;
    output Real y;

  equation
    der(x) = (K * u - x)/tau + ex.y;
    // y = delta_y*Modelica.Math.tanh(x + ey.y) + mean_y;
    y = ((y_max-y_min)/2.0)*Modelica.Math.tanh(((x + ey.y) - (y_max + y_min)/2.0)/((y_max-y_min)/2.0)) + (y_max + y_min)/2.0;
    
end SaturatedNoisyFO;
