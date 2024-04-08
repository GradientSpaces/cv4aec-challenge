%module warping_error

%include "std_string.i"
%include "std_vector.i"

using namespace std;

%template(cpp_nested1_IntVector) vector<int>;

%{
#include "warping_error.h"
%}

//double-check that this is indeed %include !!!
%include "warping_error.h"