function [strength] = vector_strength(vect)

strength = sqrt(nansum(vect.^2));