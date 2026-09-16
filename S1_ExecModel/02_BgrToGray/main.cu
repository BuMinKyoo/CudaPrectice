// 02_BgrToGray — S1 2D 인덱싱: BGR byte 버퍼 → grayscale
//
// 입력 포맷은 내 프레임 포맷 그대로: interleaved BGR, 8bit, 행 우선(HWC), pitch 없음
//   pixel(x, y) = src[(y * W + x) * 3 + {0:B, 1:G, 2:R}]
//
// 실험
//   A) CPU 결과와 바이트 단위 비교 (불일치 0 이어야 함)
//   B) 2D 블록 모양별 커널 시간 (8x8, 16x16, 32x32, 32x8, 64x4, 256x1)
//   C) 결과를 PGM/PPM 으로 저장해 눈으로 확인 (IrfanView, GIMP, VS Code 확장 등으로 열기)
//
// 사용법:  02_BgrToGray.exe [W H]   (기본 1920 1080)

#include "cu_common.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// BT.601 정수 근사:  Y = 0.114B + 0.587G + 0.299R
//   ×256 → 29, 150, 77 (합 256), +128 은 반올림
#define GRAY_B 29
#define GRAY_G 150
#define GRAY_R 77

// ---------------------------------------------------------------- 커널
__global__ void BgrToGray(const uint8_t* src, uint8_t* dst, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const int si = (y * w + x) * 3;
    const int b = src[si];
    const int g = src[si + 1];
    const int r = src[si + 2];
    dst[y * w + x] = static_cast<uint8_t>((GRAY_B * b + GRAY_G * g + GRAY_R * r + 128) >> 8);
}

// ---------------------------------------------------------------- CPU 기준
static void BgrToGrayCpu(const uint8_t* src, uint8_t* dst, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int si = (y * w + x) * 3;
            dst[y * w + x] = static_cast<uint8_t>(
                (GRAY_B * src[si] + GRAY_G * src[si + 1] + GRAY_R * src[si + 2] + 128) >> 8);
        }
    }
}

// ---------------------------------------------------------------- 테스트 이미지
// 가로 그라데이션(R) + 세로 그라데이션(G) + 체커(B). 외부 이미지 라이브러리 없이 만든다.
static std::vector<uint8_t> MakeTestBgr(int w, int h) {
    std::vector<uint8_t> img(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = (static_cast<size_t>(y) * w + x) * 3;
            img[i + 0] = (((x / 64) + (y / 64)) % 2) ? 220 : 30;   // B
            img[i + 1] = static_cast<uint8_t>(y * 255 / (h - 1));  // G
            img[i + 2] = static_cast<uint8_t>(x * 255 / (w - 1));  // R
        }
    }
    return img;
}

static void SavePpmFromBgr(const char* path, const std::vector<uint8_t>& bgr, int w, int h) {
    FILE* f = nullptr;
    if (fopen_s(&f, path, "wb") != 0 || !f) return;
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    std::vector<uint8_t> rgb(bgr.size());
    for (size_t i = 0; i < bgr.size(); i += 3) {   // PPM 은 RGB 순서
        rgb[i] = bgr[i + 2];
        rgb[i + 1] = bgr[i + 1];
        rgb[i + 2] = bgr[i];
    }
    std::fwrite(rgb.data(), 1, rgb.size(), f);
    std::fclose(f);
}

static void SavePgm(const char* path, const std::vector<uint8_t>& gray, int w, int h) {
    FILE* f = nullptr;
    if (fopen_s(&f, path, "wb") != 0 || !f) return;
    std::fprintf(f, "P5\n%d %d\n255\n", w, h);
    std::fwrite(gray.data(), 1, gray.size(), f);
    std::fclose(f);
}

int main(int argc, char** argv) {
    const int w = (argc > 2) ? std::atoi(argv[1]) : 1920;
    const int h = (argc > 2) ? std::atoi(argv[2]) : 1080;
    const size_t srcBytes = static_cast<size_t>(w) * h * 3;
    const size_t dstBytes = static_cast<size_t>(w) * h;
    const int kRepeat = 30;

    std::printf("=== 02_BgrToGray  %d x %d ===\n\n", w, h);

    const std::vector<uint8_t> bgr = MakeTestBgr(w, h);
    std::vector<uint8_t> grayCpu(dstBytes), grayGpu(dstBytes);

    // ---- CPU
    std::vector<double> cpuSamples;
    CpuTimer cpu;
    for (int r = 0; r < 5; ++r) {
        cpu.Start();
        BgrToGrayCpu(bgr.data(), grayCpu.data(), w, h);
        cpuSamples.push_back(cpu.ElapsedMs());
    }
    const double cpuMs = Median(cpuSamples);

    // ---- GPU 메모리 + 업로드
    uint8_t* dSrc = nullptr;
    uint8_t* dDst = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dSrc, srcBytes));
    CUDA_CHECK(cudaMalloc((void**)&dDst, dstBytes));

    GpuTimer t;
    t.Start();
    CUDA_CHECK(cudaMemcpy(dSrc, bgr.data(), srcBytes, cudaMemcpyHostToDevice));
    t.Stop();
    const double h2dMs = t.ElapsedMs();

    // ---- 워밍업
    {
        dim3 block(16, 16);
        dim3 grid(DivUp(w, 16), DivUp(h, 16));
        BgrToGray<<<grid, block>>>(dSrc, dDst, w, h);
        CUDA_CHECK_LAUNCH();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // =========================================================== 블록 모양 스윕
    struct Shape { int bx, by; };
    const Shape shapes[] = {{8, 8}, {16, 16}, {32, 32}, {32, 8}, {64, 4}, {256, 1}};

    std::printf("| block | threads/block | grid | kernel ms | vs CPU | mismatch |\n");
    std::printf("|---|---:|---|---:|---:|---:|\n");

    for (const Shape& s : shapes) {
        dim3 block(s.bx, s.by);
        dim3 grid(DivUp(w, s.bx), DivUp(h, s.by));

        std::vector<double> samples;
        for (int r = 0; r < kRepeat; ++r) {
            t.Start();
            BgrToGray<<<grid, block>>>(dSrc, dDst, w, h);
            CUDA_CHECK_LAUNCH();
            t.Stop();
            samples.push_back(t.ElapsedMs());
        }

        CUDA_CHECK(cudaMemcpy(grayGpu.data(), dDst, dstBytes, cudaMemcpyDeviceToHost));
        size_t mismatch = 0;
        for (size_t i = 0; i < dstBytes; ++i) mismatch += (grayCpu[i] != grayGpu[i]);

        const double ms = Median(samples);
        std::printf("| %3dx%-3d | %4d | %4ux%-4u | %7.3f | %6.1fx | %zu |\n",
                    s.bx, s.by, s.bx * s.by, grid.x, grid.y, ms, cpuMs / ms, mismatch);
    }

    t.Start();
    CUDA_CHECK(cudaMemcpy(grayGpu.data(), dDst, dstBytes, cudaMemcpyDeviceToHost));
    t.Stop();
    const double d2hMs = t.ElapsedMs();

    std::printf("\n[참고] CPU %.3f ms | H2D %.3f ms | D2H %.3f ms\n", cpuMs, h2dMs, d2hMs);

    SavePpmFromBgr("input_out.ppm", bgr, w, h);
    SavePgm("gray_out.pgm", grayGpu, w, h);
    std::printf("[저장] input_out.ppm, gray_out.pgm  (프로젝트 폴더)\n");

    CUDA_CHECK(cudaFree(dSrc));
    CUDA_CHECK(cudaFree(dDst));
    return 0;
}
