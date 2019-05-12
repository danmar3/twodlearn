if test "${JAVA_HOME}" = ""; then
  export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
fi
export JMODELICA_HOME=/home/marinodl/libraries/jmodelica2 
export IPOPT_HOME=/home/marinodl/libraries/ipopt/Ipopt-3.12.8/build/ 
export SUNDIALS_HOME=/home/marinodl/libraries/jmodelica2/ThirdParty/Sundials 
export PYTHONPATH=:/home/marinodl/libraries/jmodelica2/Python/::$PYTHONPATH 
export LD_LIBRARY_PATH=:/home/marinodl/libraries/ipopt/Ipopt-3.12.8/build//lib/:/home/marinodl/libraries/jmodelica2/ThirdParty/Sundials/lib:/home/marinodl/libraries/jmodelica2/ThirdParty/CasADi/lib:$LD_LIBRARY_PATH 
export SEPARATE_PROCESS_JVM=/usr/lib/jvm/java-8-openjdk-amd64/
