"""cffi build for a thin libdmtx shim.

We do NOT use pylibdmtx: it runs region-find + matrix-decode in one shot and
only returns the payload. We need the two stages split (to tell a *finding*
failure from a *sampling/RS* failure) plus the region geometry (to measure
pixels-per-module on every image, success or fail) plus a hard per-call
timeout (so a failing region search can't burn 60 s).

The shim returns only plain structs/buffers we control, so nothing in Python
ever touches libdmtx's internal structs -> robust across point releases.

Build:  python -m dmtxslide._build_dmtx
"""
from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef(r"""
typedef struct {
    int found;        /* region located at all                 */
    int decoded;      /* region located AND matrix decoded      */
    int symbol_rows;  /* total rows incl. alignment             */
    int symbol_cols;
    int bmin_x; int bmin_y; int bmax_x; int bmax_y;  /* region bbox in px */
    int polarity;     /* +1 dark-on-light, -1 light-on-dark     */
    int data_len;
} StageResult;

int dtmx_decode_staged(unsigned char *pxl, int w, int h,
                       int timeout_ms, int edge_thresh,
                       unsigned char *out, int out_cap,
                       StageResult *res);

int dtmx_encode(unsigned char *data, int n, int module_size, int margin,
                unsigned char *out, int out_cap, int *out_w, int *out_h);
""")

ffibuilder.set_source(
    "dmtxslide._dmtx",
    r"""
    #include <dmtx.h>
    #include <string.h>

    typedef struct {
        int found; int decoded;
        int symbol_rows; int symbol_cols;
        int bmin_x; int bmin_y; int bmax_x; int bmax_y;
        int polarity; int data_len;
    } StageResult;

    /* 8bpp grayscale in. Two stages, each observable; region search is
       bounded by timeout_ms so a miss returns fast instead of grinding. */
    int dtmx_decode_staged(unsigned char *pxl, int w, int h,
                           int timeout_ms, int edge_thresh,
                           unsigned char *out, int out_cap,
                           StageResult *res)
    {
        memset(res, 0, sizeof(*res));
        DmtxImage *img = dmtxImageCreate(pxl, w, h, DmtxPack8bppK);
        if (img == NULL) return -1;
        DmtxDecode *dec = dmtxDecodeCreate(img, 1);
        if (dec == NULL) { dmtxImageDestroy(&img); return -1; }
        if (edge_thresh > 0) dmtxDecodeSetProp(dec, DmtxPropEdgeThresh, edge_thresh);

        DmtxTime deadline = dmtxTimeAdd(dmtxTimeNow(), timeout_ms);
        DmtxRegion *reg = dmtxRegionFindNext(dec, &deadline);
        if (reg != NULL) {
            res->found       = 1;
            res->symbol_rows = reg->symbolRows;
            res->symbol_cols = reg->symbolCols;
            res->bmin_x = reg->boundMin.X; res->bmin_y = reg->boundMin.Y;
            res->bmax_x = reg->boundMax.X; res->bmax_y = reg->boundMax.Y;
            res->polarity = reg->polarity;
            DmtxMessage *msg = dmtxDecodeMatrixRegion(dec, reg, DmtxUndefined);
            if (msg != NULL) {
                res->decoded = 1;
                int n = msg->outputIdx;
                if (n > out_cap) n = out_cap;
                memcpy(out, msg->output, n);
                res->data_len = n;
                dmtxMessageDestroy(&msg);
            }
            dmtxRegionDestroy(&reg);
        }
        dmtxDecodeDestroy(&dec);
        dmtxImageDestroy(&img);
        return 0;
    }

    /* Render a payload to a clean module bitmap (8bpp). module_size=1 +
       small margin gives an almost-raw grid that Python then scales and
       degrades for the synthetic test set. */
    int dtmx_encode(unsigned char *data, int n, int module_size, int margin,
                    unsigned char *out, int out_cap, int *out_w, int *out_h)
    {
        DmtxEncode *enc = dmtxEncodeCreate();
        if (enc == NULL) return -1;
        dmtxEncodeSetProp(enc, DmtxPropModuleSize, module_size);
        dmtxEncodeSetProp(enc, DmtxPropMarginSize, margin);
        if (dmtxEncodeDataMatrix(enc, n, data) == DmtxFail) {
            dmtxEncodeDestroy(&enc); return -2;
        }
        int w   = dmtxImageGetProp(enc->image, DmtxPropWidth);
        int h   = dmtxImageGetProp(enc->image, DmtxPropHeight);
        int bpp = dmtxImageGetProp(enc->image, DmtxPropBytesPerPixel);
        *out_w = w; *out_h = h;
        if (w * h > out_cap) { dmtxEncodeDestroy(&enc); return -3; }
        for (int i = 0; i < w * h; i++) out[i] = enc->image->pxl[i * bpp];
        dmtxEncodeDestroy(&enc);
        return 0;
    }
    """,
    libraries=["dmtx"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
