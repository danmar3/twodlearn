within ;
package CC0DTdl
   model CC0D_explicit_x0
      import Plants = CombinedCycle.CombinedCycle.Optimization.Plants;
      extends Plants.CC0D;
      parameter superheater_gas_side.gas_out.T_init = 625;
      parameter evaporator_gas_side.gas_out.T_init = 500;
      parameter economizer_gas_side.gas_out.T_init = 450;
      parameter economizer.wat_liq_out.T_init = 400;
      parameter evaporator.alpha_init = 0.5;
      parameter evaporator.p_init = 3000000.0;
      parameter superheater.wat_vap_out.T_init = 550.0;
      parameter turbineShaft.T__2_init = 550.0;
      parameter PI.x_init = 0.0;
      initial equation
         // economizer.wat_liq_out.T = 459;
         // economizer_gas_side.gas_out.T= 490;
         // evaporator.p= 7.5e6;
         // evaporator_gas_side.gas_out.T=454;
         // superheater.wat_vap_out.T = 797;
         // superheater_gas_side.gas_out.T = 746;
         // turbineShaft.T__2 = 791;
         superheater_gas_side.gas_out.T = 625;
         evaporator_gas_side.gas_out.T = 500;
         economizer_gas_side.gas_out.T = 450;
         economizer.wat_liq_out.T = 400;
         evaporator.alpha = 0.5;
         evaporator.p = 3000000.0;
         superheater.wat_vap_out.T = 550.0;
         turbineShaft.T__2 = 550.0;
         PI.x = 0.0;
   end CC0D_explicit_x0;
end CC0DTdl;
