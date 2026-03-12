#include <inttypes.h>
#include <math.h>
//#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Populate array with coordinates
void find_coordinates(const uint64_t Ni, const uint64_t Nj, const uint64_t Nlabels,
                           uint64_t *coordinates, uint64_t *counts_cumsum, uint64_t *image) {

    uint64_t i, j, g, base, icx, icy;
    uint64_t* indices = calloc((Nlabels+1), sizeof(uint64_t));  // initialized with 0s

//    omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(1); // Use (X) threads for all consecutive parallel regions

    // Loop over image
//    #pragma omp parallel for collapse(2)
    for (i = 0; i < Ni; i++) {
    for (j = 0; j < Nj; j++) {

        // Global index for image
        g = i*Nj + j;

        // Get pixel label
        uint64_t label = image[g];
        if (label>1){
            //printf("i:%ld j:%ld g:%ld\n", i, j, g);
            //printf("label %ld\n", label);
            //printf("indices %ld\n", indices[label]);

            // Global index for coordinates
            base = counts_cumsum[label-1] * 2;
            icx = base + indices[label]*2;
            icy = icx + 1;
            
            //printf("icx: %ld\n", icx);
            coordinates[icx] = i;
            coordinates[icy] = j;

//            #pragma omp atomic
            indices[label] += 1;
        }
    }
    }

    free(indices);
}



// Compute object moments
void object_moments(const uint64_t Nlabels, const uint64_t Nmoments,
                           uint64_t *coordinates, uint64_t *counts_cumsum, uint64_t *all_counts, uint64_t *moments) {

    uint64_t l, j, g, base, area, icx, icy;
    uint64_t x, y;

//    omp_set_dynamic(0);     // Explicitly disable dynamic teams
//    omp_set_num_threads(1); // Use (X) threads for all consecutive parallel regions

    // Loop over labels
//    #pragma omp parallel for

    for (l=2; l<(Nlabels+1); l++) // exclude labels 0(bkg), 1(nuclei shape cells)
    {
        base = counts_cumsum[l-1] * 2;
        area = all_counts[l];
        //printf("%d %d\n", base, area);

        for (j=0; j<area; j++) {
            icx = base + j*2;
            icy = icx + 1;
            //printf("%d %d\n", icx, icy);
            x = coordinates[icx];
            y = coordinates[icy];

            //printf("%d %d %d\n", l, j, g);
            g = Nmoments * l;
            //printf("label:%d pixel:%d 6*label:%d x:%d y:%d\n", l, j, g, x, y);
            moments[g+0] += 1;       // M00
            moments[g+1] += x;       // M10
            moments[g+2] += y;       // M01
            moments[g+3] += x*y;     // M11
            moments[g+4] += x*x;     // M20
            moments[g+5] += y*y;     // M02
            //printf("%d: %d %d %d %d\n", j, x, y, x*x, moments[g+4]);
        }
    }
}
