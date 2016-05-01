// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.
//
// Copyright (C) 2012-2016, Jordan Melo
//
//      Authority FX, Inc.
//      www.authorityfx.com

#include "gauss_blur.h"

#include "image.h"
#include "threading.h"

namespace afx {

void GaussianBlur::Process(Image* in, Image* out, float blurRadius) {
    Image temp_image(in->GetSize().width, in->GetSize().height);

    float sigma = blurRadius;
    int kernel_size = (int) ceil(7 * sigma) | 1;
    int kernel_center = kernel_size / 2;

    float normalizing_factor = 1.0f / (sqrt(6.28318530717958647f) * sigma);
    float sigma_inv_sqr = 1.0f / (sigma * sigma);

    float sum = 0.0f;

    std::vector<float> kernel(kernel_size);

    for (int i = 0; i < kernel_size; i++) {
        kernel[i] = normalizing_factor * exp(-0.5f * (i - kernel_center) * (i - kernel_center) * sigma_inv_sqr);
        sum += kernel[i];
    }
    float invSum = 1.0f / sum;
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] *= invSum;
    }

    RunChunks(boost::bind(&GaussianBlur::HorizontalGaussianBlur, this, _1, in, &temp_image, &kernel[0], kernel_size));
    RunChunks(boost::bind(&GaussianBlur::VerticalGaussianBlur, this, _1, &temp_image, out, &kernel[0], kernel_size));
}
void GaussianBlur::HorizontalGaussianBlur(Bounds region, Image* in, Image* out, float* kernel, int kernel_size) {
    float* po;

    int kernel_center = kernel_size / 2;
    int xx;

    int width = in->GetSize().width;

    for (int y = region.y1; y <= region.y2; y++) {
        po = out->GetPtr(region.x1, y);
        for (int x = region.x1; x <= region.x2; x++) {
            *po = 0.0f;
            for (int i = 0; i < kernel_size; i++) {
                xx = x - kernel_center + i;
                if (xx < 0) xx = 0;
                if (xx > width - 1) xx = width - 1;
                *po += *in->GetPtr(xx, y) * kernel[i];
            }
            po++;
        }
    }
}
void GaussianBlur::VerticalGaussianBlur(Bounds region, Image* in, Image* out, float* kernel, int kernel_size) {
    float* po;
    int kernel_center = kernel_size / 2;
    int yy;
    int height = in->GetSize().height;

    for (int y = region.y1; y <= region.y2; y++) {
        po = out->GetPtr(region.x1, y);
        for (int x = region.x1; x <= region.x2; x++) {
            *po = 0.0f;
            for (int i = 0; i < kernel_size; i++) {
                yy = y - kernel_center + i;
                if (yy < 0) yy = 0;
                if (yy > height - 1) yy = height - 1;
                *po += *in->GetPtr(x, yy) * kernel[i];
            }
            po++;
        }
    }
}

} // namespace afx
