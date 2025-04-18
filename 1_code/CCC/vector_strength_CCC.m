function [strength] = vector_strength_CCC(vect)

%strength = sqrt(nansum(vect.^2));
strength = abs(nanmean(exp(1i * vect)));