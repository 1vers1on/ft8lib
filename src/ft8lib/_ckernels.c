/* C kernels for the decoder hot paths.
 *
 * Direct transcriptions of the WSJT-X Fortran inner loops
 * (bpdecode174_91.f90, osd174_91.f90, sync8d.f90) and of the WSPR
 * demodulator/decoder (lib/wsprd/wsprd.c sync_and_demodulate and
 * noncoherent_sequence_detection, lib/wsprd/fano.c) exposed to Python.
 * The pure-numpy implementations in ldpc.py / decode.py / wspr.py remain
 * as the fallback when this module is unavailable.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#define LDPC_N 174 /* codeword bits */
#define LDPC_M 83  /* parity checks */
#define LDPC_K 91  /* message bits (77 + CRC14) */

#define CRC_POLY 0x2757u
#define FT8LIB_PI 3.14159265358979323846

/* CRC-14 check of the first 91 bits of a codeword (port of crc.py). */
static int
crc14_check_bits(const uint8_t *cw)
{
    unsigned reg = 0;
    int i;
    for (i = 0; i < 77; i++) {
        reg = (reg << 1) | (cw[i] ? 1u : 0u);
        if (reg & (1u << 14))
            reg ^= (1u << 14) | CRC_POLY;
    }
    for (i = 0; i < 19; i++) { /* 5 pad bits + 14 augmentation zeros */
        reg <<= 1;
        if (reg & (1u << 14))
            reg ^= (1u << 14) | CRC_POLY;
    }
    reg &= 0x3FFFu;
    {
        unsigned expected = 0;
        for (i = 77; i < 91; i++)
            expected = (expected << 1) | (cw[i] ? 1u : 0u);
        return reg == (unsigned)expected;
    }
}

/* Fetch a C-contiguous array of the given type/ndim or set an error. */
static PyArrayObject *
as_array(PyObject *obj, int typenum, int ndim, const char *name)
{
    PyArrayObject *arr =
        (PyArrayObject *)PyArray_FROM_OTF(obj, typenum, NPY_ARRAY_IN_ARRAY);
    if (arr == NULL)
        return NULL;
    if (PyArray_NDIM(arr) != ndim) {
        PyErr_Format(PyExc_ValueError, "%s: expected %d-d array", name, ndim);
        Py_DECREF(arr);
        return NULL;
    }
    return arr;
}

/* bp_hybrid(llr, ap_free, nm, nrw, mn, edge_slot, max_iterations, max_osd)
 *   -> (status, cw, zsave, nsave)
 *
 * Belief-propagation loop shared by bp_decode and decode174_91.  status
 * 1 = codeword with valid CRC (in cw), -1 = stagnated early, 0 = iterations
 * exhausted.  zsave holds the accumulated LLR sums of iterations
 * 1..max_osd for OSD reprocessing.
 */
static PyObject *
bp_hybrid(PyObject *self, PyObject *args)
{
    PyObject *llr_o, *ap_o, *nm_o, *nrw_o, *mn_o, *slot_o;
    long max_iterations, max_osd;
    PyArrayObject *llr_a = NULL, *ap_a = NULL, *nm_a = NULL, *nrw_a = NULL,
                  *mn_a = NULL, *slot_a = NULL, *cw_a = NULL, *zsave_a = NULL;

    if (!PyArg_ParseTuple(args, "OOOOOOll", &llr_o, &ap_o, &nm_o, &nrw_o,
                          &mn_o, &slot_o, &max_iterations, &max_osd))
        return NULL;

    llr_a = as_array(llr_o, NPY_DOUBLE, 1, "llr");
    ap_a = as_array(ap_o, NPY_UINT8, 1, "ap_free");
    nm_a = as_array(nm_o, NPY_INT64, 2, "nm");
    nrw_a = as_array(nrw_o, NPY_INT64, 1, "nrw");
    mn_a = as_array(mn_o, NPY_INT64, 2, "mn");
    slot_a = as_array(slot_o, NPY_INT64, 2, "edge_slot");
    if (!llr_a || !ap_a || !nm_a || !nrw_a || !mn_a || !slot_a)
        goto fail;
    if (PyArray_DIM(llr_a, 0) != LDPC_N || PyArray_DIM(ap_a, 0) != LDPC_N ||
        PyArray_DIM(nm_a, 0) != LDPC_M || PyArray_DIM(nm_a, 1) != 7 ||
        PyArray_DIM(nrw_a, 0) != LDPC_M || PyArray_DIM(mn_a, 0) != LDPC_N ||
        PyArray_DIM(mn_a, 1) != 3 || PyArray_DIM(slot_a, 0) != LDPC_N ||
        PyArray_DIM(slot_a, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, "bp_hybrid: bad array shape");
        goto fail;
    }

    {
        const double *llr = (const double *)PyArray_DATA(llr_a);
        const uint8_t *ap_free = (const uint8_t *)PyArray_DATA(ap_a);
        const int64_t *nm = (const int64_t *)PyArray_DATA(nm_a);
        const int64_t *nrw = (const int64_t *)PyArray_DATA(nrw_a);
        const int64_t *mn = (const int64_t *)PyArray_DATA(mn_a);
        const int64_t *slot = (const int64_t *)PyArray_DATA(slot_a);

        npy_intp nsavemax = max_osd > 0 ? (npy_intp)max_osd : 0;
        npy_intp cw_dims[1] = {LDPC_N};
        npy_intp zs_dims[2] = {nsavemax, LDPC_N};
        uint8_t *cw;
        double *zsave;
        long nsave = 0;
        int status = 0;

        double tov[LDPC_N][3], toc[LDPC_M][7], tanhtoc[LDPC_M][7];
        double zn[LDPC_N], zsum[LDPC_N];
        int ncnt = 0, nclast = 0;
        long iteration;
        int i, j, k, s;

        cw_a = (PyArrayObject *)PyArray_ZEROS(1, cw_dims, NPY_UINT8, 0);
        zsave_a = (PyArrayObject *)PyArray_ZEROS(2, zs_dims, NPY_DOUBLE, 0);
        if (!cw_a || !zsave_a)
            goto fail;
        cw = (uint8_t *)PyArray_DATA(cw_a);
        zsave = (double *)PyArray_DATA(zsave_a);

        memset(tov, 0, sizeof(tov));
        memset(zsum, 0, sizeof(zsum));

        Py_BEGIN_ALLOW_THREADS;
        for (iteration = 0; iteration <= max_iterations; iteration++) {
            int ncheck = 0;

            for (j = 0; j < LDPC_N; j++) {
                if (ap_free[j])
                    zn[j] = llr[j] + tov[j][0] + tov[j][1] + tov[j][2];
                else
                    zn[j] = llr[j];
                zsum[j] += zn[j];
                cw[j] = zn[j] > 0.0 ? 1 : 0;
            }
            if (iteration >= 1 && iteration <= nsavemax) {
                memcpy(zsave + (iteration - 1) * LDPC_N, zsum,
                       sizeof(zsum));
                nsave = iteration;
            }

            for (i = 0; i < LDPC_M; i++) {
                int syn = 0;
                for (k = 0; k < nrw[i]; k++)
                    syn += cw[nm[i * 7 + k]];
                if (syn % 2 != 0)
                    ncheck++;
            }
            if (ncheck == 0 && crc14_check_bits(cw)) {
                status = 1;
                break;
            }

            if (iteration > 0) {
                if (ncheck - nclast < 0)
                    ncnt = 0;
                else
                    ncnt++;
                if (ncnt >= 5 && iteration >= 10 && ncheck > 15) {
                    status = -1;
                    break;
                }
            }
            nclast = ncheck;

            /* bit-to-check: total belief minus what that check gave */
            for (j = 0; j < LDPC_N; j++) {
                for (s = 0; s < 3; s++)
                    toc[mn[j * 3 + s]][slot[j * 3 + s]] = zn[j] - tov[j][s];
            }
            for (i = 0; i < LDPC_M; i++) {
                for (k = 0; k < nrw[i]; k++)
                    tanhtoc[i][k] = tanh(-toc[i][k] / 2.0);
            }

            /* check-to-bit: product over the check excluding the bit */
            for (j = 0; j < LDPC_N; j++) {
                for (s = 0; s < 3; s++) {
                    int64_t chk = mn[j * 3 + s];
                    int64_t sl = slot[j * 3 + s];
                    double p = 1.0, t;
                    for (k = 0; k < nrw[chk]; k++) {
                        if (k != sl)
                            p *= tanhtoc[chk][k];
                    }
                    t = -p;
                    if (t > 0.9999999999)
                        t = 0.9999999999;
                    else if (t < -0.9999999999)
                        t = -0.9999999999;
                    tov[j][s] = 2.0 * atanh(t);
                }
            }
        }
        Py_END_ALLOW_THREADS;

        Py_DECREF(llr_a);
        Py_DECREF(ap_a);
        Py_DECREF(nm_a);
        Py_DECREF(nrw_a);
        Py_DECREF(mn_a);
        Py_DECREF(slot_a);
        return Py_BuildValue("iNNl", status, cw_a, zsave_a, nsave);
    }

fail:
    Py_XDECREF(llr_a);
    Py_XDECREF(ap_a);
    Py_XDECREF(nm_a);
    Py_XDECREF(nrw_a);
    Py_XDECREF(mn_a);
    Py_XDECREF(slot_a);
    Py_XDECREF(cw_a);
    Py_XDECREF(zsave_a);
    return NULL;
}

/* Gaussian elimination for OSD: identity on the first K columns of the
 * (K, N) uint8 generator, swapping in later columns when necessary and
 * permuting `indices` to match.  0 in the degenerate no-pivot case. */
static int
ge_reduce(uint8_t *g, int64_t *indices)
{
    int d, r, c, col;

    for (d = 0; d < LDPC_K; d++) {
        int pivot = -1;
        for (col = d; col < LDPC_N; col++) {
            if (g[d * LDPC_N + col]) {
                pivot = col;
                break;
            }
        }
        if (pivot < 0)
            return 0;
        if (pivot != d) {
            int64_t itmp = indices[d];
            indices[d] = indices[pivot];
            indices[pivot] = itmp;
            for (r = 0; r < LDPC_K; r++) {
                uint8_t tmp = g[r * LDPC_N + d];
                g[r * LDPC_N + d] = g[r * LDPC_N + pivot];
                g[r * LDPC_N + pivot] = tmp;
            }
        }
        for (r = 0; r < LDPC_K; r++) {
            if (r != d && g[r * LDPC_N + d]) {
                uint8_t *dst = g + r * LDPC_N;
                const uint8_t *src = g + d * LDPC_N;
                for (c = 0; c < LDPC_N; c++)
                    dst[c] ^= src[c];
            }
        }
    }
    return 1;
}

/* osd_ge(genmrb, indices) -> bool: in-place Gaussian elimination. */
static PyObject *
osd_ge(PyObject *self, PyObject *args)
{
    PyObject *g_o, *idx_o;
    PyArrayObject *g_a, *idx_a;

    if (!PyArg_ParseTuple(args, "OO", &g_o, &idx_o))
        return NULL;
    if (!PyArray_Check(g_o) || !PyArray_Check(idx_o)) {
        PyErr_SetString(PyExc_TypeError, "osd_ge: expected ndarrays");
        return NULL;
    }
    g_a = (PyArrayObject *)g_o;
    idx_a = (PyArrayObject *)idx_o;
    if (PyArray_TYPE(g_a) != NPY_UINT8 || PyArray_NDIM(g_a) != 2 ||
        PyArray_DIM(g_a, 0) != LDPC_K || PyArray_DIM(g_a, 1) != LDPC_N ||
        !PyArray_IS_C_CONTIGUOUS(g_a) || !PyArray_ISWRITEABLE(g_a) ||
        PyArray_TYPE(idx_a) != NPY_INT64 || PyArray_NDIM(idx_a) != 1 ||
        PyArray_DIM(idx_a, 0) != LDPC_N ||
        !PyArray_IS_C_CONTIGUOUS(idx_a) || !PyArray_ISWRITEABLE(idx_a)) {
        PyErr_SetString(PyExc_ValueError,
                        "osd_ge: need writable C-contiguous uint8 (91,174) "
                        "and int64 (174,) arrays");
        return NULL;
    }

    if (ge_reduce((uint8_t *)PyArray_DATA(g_a),
                  (int64_t *)PyArray_DATA(idx_a)))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/* osd(llr, apmask, gfull, order, norder) -> (found, cw, nhardmin, dmin)
 *
 * Full ordered-statistics decode (port of the osd_decode body in ldpc.py):
 * permute the generator columns by decreasing reliability (`order`), reduce
 * to systematic form, and try all error patterns of weight <= norder over
 * the 91 most-reliable-basis positions.  found is 1 when the best codeword
 * has a valid CRC.
 */
static PyObject *
osd(PyObject *self, PyObject *args)
{
    PyObject *llr_o, *apm_o, *g_o, *order_o;
    long norder;
    PyArrayObject *llr_a = NULL, *apm_a = NULL, *g_a = NULL,
                  *order_a = NULL, *cw_a = NULL;

    if (!PyArg_ParseTuple(args, "OOOOl", &llr_o, &apm_o, &g_o, &order_o,
                          &norder))
        return NULL;
    llr_a = as_array(llr_o, NPY_DOUBLE, 1, "llr");
    apm_a = as_array(apm_o, NPY_UINT8, 1, "apmask");
    g_a = as_array(g_o, NPY_UINT8, 2, "gfull");
    order_a = as_array(order_o, NPY_INT64, 1, "order");
    if (!llr_a || !apm_a || !g_a || !order_a)
        goto fail;
    if (PyArray_DIM(llr_a, 0) != LDPC_N || PyArray_DIM(apm_a, 0) != LDPC_N ||
        PyArray_DIM(g_a, 0) != LDPC_K || PyArray_DIM(g_a, 1) != LDPC_N ||
        PyArray_DIM(order_a, 0) != LDPC_N) {
        PyErr_SetString(PyExc_ValueError, "osd: bad array shape");
        goto fail;
    }

    {
        const double *rx = (const double *)PyArray_DATA(llr_a);
        const uint8_t *apm = (const uint8_t *)PyArray_DATA(apm_a);
        const uint8_t *gfull = (const uint8_t *)PyArray_DATA(g_a);
        const int64_t *order = (const int64_t *)PyArray_DATA(order_a);

        npy_intp cw_dims[1] = {LDPC_N};
        uint8_t *cw;
        uint8_t hdec[LDPC_N], hdec_p[LDPC_N], apm_p[LDPC_N];
        uint8_t c0[LDPC_N], cw_p[LDPC_N];
        double absrx[LDPC_N], absrx_p[LDPC_N], w[LDPC_N];
        int64_t indices[LDPC_N];
        static uint8_t genmrb[LDPC_K][LDPC_N]; /* scratch; GIL held */
        static double R[LDPC_K][LDPC_N], Rw[LDPC_K][LDPC_N];
        double w2, cst, best_dd;
        long nhardmin = 0;
        double dmin = 0.0;
        int found, p1 = -1, p2 = -1;
        int i, j, n;

        cw_a = (PyArrayObject *)PyArray_ZEROS(1, cw_dims, NPY_UINT8, 0);
        if (!cw_a)
            goto fail;
        cw = (uint8_t *)PyArray_DATA(cw_a);

        for (n = 0; n < LDPC_N; n++) {
            hdec[n] = rx[n] >= 0.0 ? 1 : 0;
            absrx[n] = fabs(rx[n]);
            indices[n] = order[n];
        }
        for (i = 0; i < LDPC_K; i++) {
            for (n = 0; n < LDPC_N; n++)
                genmrb[i][n] = gfull[i * LDPC_N + order[n]];
        }

        if (!ge_reduce(&genmrb[0][0], indices)) {
            /* degenerate; should not happen */
            memcpy(cw, hdec, LDPC_N);
            Py_DECREF(llr_a);
            Py_DECREF(apm_a);
            Py_DECREF(g_a);
            Py_DECREF(order_a);
            return Py_BuildValue("iNld", 0, cw_a, (long)-1, 0.0);
        }

        for (n = 0; n < LDPC_N; n++) {
            hdec_p[n] = hdec[indices[n]];
            absrx_p[n] = absrx[indices[n]];
            apm_p[n] = apm[indices[n]];
        }

        /* order-0 codeword from the K most reliable hard decisions */
        memset(c0, 0, sizeof(c0));
        for (i = 0; i < LDPC_K; i++) {
            if (hdec_p[i]) {
                for (n = 0; n < LDPC_N; n++)
                    c0[n] ^= genmrb[i][n];
            }
        }

        /* dd(S) = const - 0.5 * B.(prod_{i in S} R_i) with B = absrx * +/-1
         * error form; R rows in +/-1 form. */
        w2 = 0.0;
        cst = 0.0;
        for (n = 0; n < LDPC_N; n++) {
            w[n] = absrx_p[n] * (1.0 - 2.0 * (double)(c0[n] ^ hdec_p[n]));
            w2 += w[n];
            cst += absrx_p[n];
        }
        cst *= 0.5;
        best_dd = cst - 0.5 * w2;

        for (i = 0; i < LDPC_K; i++) {
            for (n = 0; n < LDPC_N; n++) {
                R[i][n] = 1.0 - 2.0 * (double)genmrb[i][n];
                Rw[i][n] = R[i][n] * w[n];
            }
        }

        if (norder >= 1) {
            for (i = 0; i < LDPC_K; i++) {
                double s = 0.0, dd1;
                if (apm_p[i])
                    continue;
                for (n = 0; n < LDPC_N; n++)
                    s += Rw[i][n];
                dd1 = cst - 0.5 * s;
                if (dd1 < best_dd) {
                    best_dd = dd1;
                    p1 = i;
                    p2 = -1;
                }
            }
        }
        if (norder >= 2) {
            for (i = 0; i < LDPC_K; i++) {
                if (apm_p[i])
                    continue;
                for (j = i + 1; j < LDPC_K; j++) {
                    double s = 0.0, dd2;
                    if (apm_p[j])
                        continue;
                    for (n = 0; n < LDPC_N; n++)
                        s += Rw[i][n] * R[j][n];
                    dd2 = cst - 0.5 * s;
                    if (dd2 < best_dd) {
                        best_dd = dd2;
                        p1 = i;
                        p2 = j;
                    }
                }
            }
        }

        memcpy(cw_p, c0, sizeof(c0));
        if (p1 >= 0) {
            for (n = 0; n < LDPC_N; n++)
                cw_p[n] ^= genmrb[p1][n];
        }
        if (p2 >= 0) {
            for (n = 0; n < LDPC_N; n++)
                cw_p[n] ^= genmrb[p2][n];
        }
        for (n = 0; n < LDPC_N; n++)
            cw[indices[n]] = cw_p[n];
        for (n = 0; n < LDPC_N; n++) {
            if (cw[n] != hdec[n]) {
                nhardmin++;
                dmin += absrx[n];
            }
        }
        found = crc14_check_bits(cw);

        Py_DECREF(llr_a);
        Py_DECREF(apm_a);
        Py_DECREF(g_a);
        Py_DECREF(order_a);
        return Py_BuildValue("iNld", found, cw_a, nhardmin, dmin);
    }

fail:
    Py_XDECREF(llr_a);
    Py_XDECREF(apm_a);
    Py_XDECREF(g_a);
    Py_XDECREF(order_a);
    Py_XDECREF(cw_a);
    return NULL;
}

/* sync8d(cd0, i0, cc, np2) -> float
 *
 * Sync power of a downsampled FT8 signal (sync8d.f90).  cc is the
 * conjugated (and optionally frequency-tweaked) Costas waveform set (7, 32).
 */
static PyObject *
sync8d(PyObject *self, PyObject *args)
{
    PyObject *cd0_o, *cc_o;
    long i0, np2;
    PyArrayObject *cd0_a = NULL, *cc_a = NULL;
    double sync = 0.0;

    if (!PyArg_ParseTuple(args, "OlOl", &cd0_o, &i0, &cc_o, &np2))
        return NULL;
    cd0_a = as_array(cd0_o, NPY_CDOUBLE, 1, "cd0");
    cc_a = as_array(cc_o, NPY_CDOUBLE, 2, "cc");
    if (!cd0_a || !cc_a)
        goto fail;
    if (PyArray_DIM(cc_a, 0) != 7 || PyArray_DIM(cc_a, 1) != 32 ||
        PyArray_DIM(cd0_a, 0) < np2) {
        PyErr_SetString(PyExc_ValueError, "sync8d: bad array shape");
        goto fail;
    }

    {
        const double *cd0 = (const double *)PyArray_DATA(cd0_a);
        const double *cc = (const double *)PyArray_DATA(cc_a);
        int i, b, k;

        for (i = 0; i < 7; i++) {
            for (b = 0; b < 3; b++) {
                long i1 = i0 + (i + 36 * b) * 32;
                if (i1 >= 0 && i1 + 31 <= np2 - 1) {
                    double zr = 0.0, zi = 0.0;
                    const double *x = cd0 + 2 * i1;
                    const double *y = cc + 2 * (i * 32);
                    for (k = 0; k < 32; k++) {
                        double xr = x[2 * k], xi = x[2 * k + 1];
                        double yr = y[2 * k], yi = y[2 * k + 1];
                        zr += xr * yr - xi * yi;
                        zi += xr * yi + xi * yr;
                    }
                    sync += zr * zr + zi * zi;
                }
            }
        }
    }
    Py_DECREF(cd0_a);
    Py_DECREF(cc_a);
    return PyFloat_FromDouble(sync);

fail:
    Py_XDECREF(cd0_a);
    Py_XDECREF(cc_a);
    return NULL;
}

/* ft4_sync_search(cd2, istarts, idfs, dt_eff, templates, offsets)
 *   -> (best_sync, best_istart, best_idf)
 *
 * Grid search of FT4 sync power over frequency offsets (idfs, Hz) and
 * block start samples (istarts), port of _ft4_sync_search in decode.py.
 * templates is the (nblocks, n64) stride-2 Costas sync waveform set;
 * offsets are the block start offsets (in stride-2 samples) within one
 * FT4 message.  Mirrors the per-idf argmax-then-compare logic exactly,
 * including its first-occurrence tie-breaking.
 */
static PyObject *
ft4_sync_search(PyObject *self, PyObject *args)
{
    PyObject *cd2_o, *istarts_o, *idfs_o, *templates_o, *offsets_o;
    double dt_eff;
    PyArrayObject *cd2_a = NULL, *istarts_a = NULL, *idfs_a = NULL,
                  *templates_a = NULL, *offsets_a = NULL;

    if (!PyArg_ParseTuple(args, "OOOdOO", &cd2_o, &istarts_o, &idfs_o,
                          &dt_eff, &templates_o, &offsets_o))
        return NULL;

    cd2_a = as_array(cd2_o, NPY_CDOUBLE, 1, "cd2");
    istarts_a = as_array(istarts_o, NPY_INT64, 1, "istarts");
    idfs_a = as_array(idfs_o, NPY_INT64, 1, "idfs");
    templates_a = as_array(templates_o, NPY_CDOUBLE, 2, "templates");
    offsets_a = as_array(offsets_o, NPY_INT64, 1, "offsets");
    if (!cd2_a || !istarts_a || !idfs_a || !templates_a || !offsets_a)
        goto fail;

    {
        const npy_intp ndmax = PyArray_DIM(cd2_a, 0);
        const npy_intp n_istart = PyArray_DIM(istarts_a, 0);
        const npy_intp n_idf = PyArray_DIM(idfs_a, 0);
        const npy_intp nblocks = PyArray_DIM(templates_a, 0);
        const npy_intp n64 = PyArray_DIM(templates_a, 1);

        if (PyArray_DIM(offsets_a, 0) != nblocks) {
            PyErr_SetString(PyExc_ValueError,
                            "ft4_sync_search: offsets/templates mismatch");
            goto fail;
        }

        {
            const double *cd2 = (const double *)PyArray_DATA(cd2_a);
            const int64_t *istarts = (const int64_t *)PyArray_DATA(istarts_a);
            const int64_t *idfs = (const int64_t *)PyArray_DATA(idfs_a);
            const double *templ = (const double *)PyArray_DATA(templates_a);
            const int64_t *offsets = (const int64_t *)PyArray_DATA(offsets_a);

            double best_val = -1.0;
            int64_t best_istart = 0, best_idf = 0;
            double *cr = PyMem_Malloc(n64 * sizeof(double));
            double *ci = PyMem_Malloc(n64 * sizeof(double));
            npy_intp fi, si, b, k;

            if (!cr || !ci) {
                PyMem_Free(cr);
                PyMem_Free(ci);
                PyErr_NoMemory();
                goto fail;
            }

            Py_BEGIN_ALLOW_THREADS;
            for (fi = 0; fi < n_idf; fi++) {
                int64_t idf = idfs[fi];
                double row_best = -1.0;
                int64_t row_best_istart = 0;

                for (k = 0; k < n64; k++) {
                    double phi = 2.0 * FT8LIB_PI * (double)idf * dt_eff * (double)k;
                    cr[k] = cos(phi);
                    ci[k] = -sin(phi);
                }

                for (si = 0; si < n_istart; si++) {
                    int64_t istart = istarts[si];
                    double sync = 0.0;

                    for (b = 0; b < nblocks; b++) {
                        int64_t i1 = istart + offsets[b];
                        if (i1 < 0 || i1 + 2 * n64 - 1 > ndmax - 1)
                            continue;
                        {
                            double zr = 0.0, zi = 0.0;
                            const double *tb = templ + 2 * b * n64;
                            for (k = 0; k < n64; k++) {
                                int64_t idx = i1 + 2 * k;
                                double xr = cd2[2 * idx], xi = cd2[2 * idx + 1];
                                double tr = tb[2 * k], ti = tb[2 * k + 1];
                                double combined_re = tr * cr[k] + ti * ci[k];
                                double combined_im = tr * ci[k] - ti * cr[k];
                                zr += xr * combined_re - xi * combined_im;
                                zi += xr * combined_im + xi * combined_re;
                            }
                            sync += sqrt(zr * zr + zi * zi);
                        }
                    }
                    if (sync > row_best) {
                        row_best = sync;
                        row_best_istart = istart;
                    }
                }
                if (row_best > best_val) {
                    best_val = row_best;
                    best_istart = row_best_istart;
                    best_idf = idf;
                }
            }
            Py_END_ALLOW_THREADS;

            PyMem_Free(cr);
            PyMem_Free(ci);

            Py_DECREF(cd2_a);
            Py_DECREF(istarts_a);
            Py_DECREF(idfs_a);
            Py_DECREF(templates_a);
            Py_DECREF(offsets_a);
            return Py_BuildValue("dLL", best_val, best_istart, best_idf);
        }
    }

fail:
    Py_XDECREF(cd2_a);
    Py_XDECREF(istarts_a);
    Py_XDECREF(idfs_a);
    Py_XDECREF(templates_a);
    Py_XDECREF(offsets_a);
    return NULL;
}

/* crc14_check(bits91) -> bool: check a 91-bit message+CRC block. */
static PyObject *
crc14_check(PyObject *self, PyObject *args)
{
    PyObject *o;
    PyArrayObject *a;
    int ok;

    if (!PyArg_ParseTuple(args, "O", &o))
        return NULL;
    a = as_array(o, NPY_UINT8, 1, "bits91");
    if (a == NULL)
        return NULL;
    if (PyArray_DIM(a, 0) != LDPC_K) {
        PyErr_SetString(PyExc_ValueError, "crc14_check: expected 91 bits");
        Py_DECREF(a);
        return NULL;
    }
    ok = crc14_check_bits((const uint8_t *)PyArray_DATA(a));
    Py_DECREF(a);
    return PyBool_FromLong(ok);
}

/* ------------------------------------------------------------------------
 * WSPR kernels (ports of lib/wsprd/wsprd.c and lib/wsprd/fano.c)
 * ------------------------------------------------------------------------ */

#define WSPR_NSYM 162
#define WSPR_NSPS 256          /* samples per symbol at 375 S/s */
#define WSPR_POLY1 0xf2d05351u /* Layland-Lushbaugh K=32 r=1/2 code */
#define WSPR_POLY2 0xe4613c47u

/* 162-bit pseudo-random sync vector (pr3 in wsprd.c) */
static const unsigned char wspr_pr3[WSPR_NSYM] = {
    1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0,
    0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,
    0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
    1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
    0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
    0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0,
    0, 0};

/* Per-symbol correlations of the baseband signal against the four FSK
 * tones, with the drift model of wsprd.c.  For each symbol i fills
 * zr/zi[tone][i]; the tone mixing tables are recomputed only when the
 * symbol frequency changes (drift != 0), like the fplast cache in C.
 * When cf/sf are non-NULL also stores each tone's per-symbol phase
 * advance factor (for block demodulation).
 */
static void
wspr_tone_corr(const double *cd, npy_intp npts, double f0, double drift,
               long lag, double zr[4][WSPR_NSYM], double zi[4][WSPR_NSYM],
               double cf[4][WSPR_NSYM], double sf[4][WSPR_NSYM])
{
    const double dt = 1.0 / 375.0, df = 375.0 / 256.0;
    const double twopidt = 2.0 * FT8LIB_PI * dt;
    const double df15 = df * 1.5, df05 = df * 0.5;
    double c0[257], s0[257], c1[257], s1[257];
    double c2[257], s2[257], c3[257], s3[257];
    double fplast = -10000.0;
    int i, j, t;
    long k;

    for (i = 0; i < WSPR_NSYM; i++) {
        double fp = f0 + (drift / 2.0) * ((double)i - 81.0) / 81.0;
        if (i == 0 || fp != fplast) {
            double dphi0 = twopidt * (fp - df15);
            double dphi1 = twopidt * (fp - df05);
            double dphi2 = twopidt * (fp + df05);
            double dphi3 = twopidt * (fp + df15);
            double cd0 = cos(dphi0), sd0 = sin(dphi0);
            double cd1 = cos(dphi1), sd1 = sin(dphi1);
            double cd2 = cos(dphi2), sd2 = sin(dphi2);
            double cd3 = cos(dphi3), sd3 = sin(dphi3);
            c0[0] = 1; s0[0] = 0; c1[0] = 1; s1[0] = 0;
            c2[0] = 1; s2[0] = 0; c3[0] = 1; s3[0] = 0;
            for (j = 1; j < 257; j++) {
                c0[j] = c0[j - 1] * cd0 - s0[j - 1] * sd0;
                s0[j] = c0[j - 1] * sd0 + s0[j - 1] * cd0;
                c1[j] = c1[j - 1] * cd1 - s1[j - 1] * sd1;
                s1[j] = c1[j - 1] * sd1 + s1[j - 1] * cd1;
                c2[j] = c2[j - 1] * cd2 - s2[j - 1] * sd2;
                s2[j] = c2[j - 1] * sd2 + s2[j - 1] * cd2;
                c3[j] = c3[j - 1] * cd3 - s3[j - 1] * sd3;
                s3[j] = c3[j - 1] * sd3 + s3[j - 1] * cd3;
            }
            fplast = fp;
        }
        if (cf != NULL) {
            cf[0][i] = c0[256]; sf[0][i] = s0[256];
            cf[1][i] = c1[256]; sf[1][i] = s1[256];
            cf[2][i] = c2[256]; sf[2][i] = s2[256];
            cf[3][i] = c3[256]; sf[3][i] = s3[256];
        }
        for (t = 0; t < 4; t++) {
            zr[t][i] = 0.0;
            zi[t][i] = 0.0;
        }
        for (j = 0; j < 256; j++) {
            k = lag + (long)i * 256 + j;
            if (k > 0 && k < npts) {
                double id = cd[2 * k], qd = cd[2 * k + 1];
                zr[0][i] += id * c0[j] + qd * s0[j];
                zi[0][i] += -id * s0[j] + qd * c0[j];
                zr[1][i] += id * c1[j] + qd * s1[j];
                zi[1][i] += -id * s1[j] + qd * c1[j];
                zr[2][i] += id * c2[j] + qd * s2[j];
                zi[2][i] += -id * s2[j] + qd * c2[j];
                zr[3][i] += id * c3[j] + qd * s3[j];
                zi[3][i] += -id * s3[j] + qd * c3[j];
            }
        }
    }
}

/* wspr_sync_demod(c, f1, ifmin, ifmax, fstep, lagmin, lagmax, lagstep,
 *                 drift1) -> (sync, freq, shift)
 *
 * Modes 0/1 of sync_and_demodulate in wsprd.c: search the given lag and
 * frequency grids for the best sync-vector correlation.
 */
static PyObject *
wspr_sync_demod(PyObject *self, PyObject *args)
{
    PyObject *c_o;
    double f1, fstep, drift1;
    long ifmin, ifmax, lagmin, lagmax, lagstep;
    PyArrayObject *c_a;

    if (!PyArg_ParseTuple(args, "Odlldllld", &c_o, &f1, &ifmin, &ifmax,
                          &fstep, &lagmin, &lagmax, &lagstep, &drift1))
        return NULL;
    c_a = as_array(c_o, NPY_CDOUBLE, 1, "c");
    if (!c_a)
        return NULL;

    {
        const double *cd = (const double *)PyArray_DATA(c_a);
        const npy_intp npts = PyArray_DIM(c_a, 0);
        double zr[4][WSPR_NSYM], zi[4][WSPR_NSYM];
        double syncmax = -1e30, fbest = f1;
        long best_shift = lagmin, lag, ifreq;
        int i;

        Py_BEGIN_ALLOW_THREADS;
        for (ifreq = ifmin; ifreq <= ifmax; ifreq++) {
            double f0 = f1 + ifreq * fstep;
            for (lag = lagmin; lag <= lagmax; lag += lagstep) {
                double ss = 0.0, totp = 0.0;
                wspr_tone_corr(cd, npts, f0, drift1, lag, zr, zi, NULL, NULL);
                for (i = 0; i < WSPR_NSYM; i++) {
                    double p0 = sqrt(zr[0][i] * zr[0][i] + zi[0][i] * zi[0][i]);
                    double p1 = sqrt(zr[1][i] * zr[1][i] + zi[1][i] * zi[1][i]);
                    double p2 = sqrt(zr[2][i] * zr[2][i] + zi[2][i] * zi[2][i]);
                    double p3 = sqrt(zr[3][i] * zr[3][i] + zi[3][i] * zi[3][i]);
                    double cmet = (p1 + p3) - (p0 + p2);
                    totp += p0 + p1 + p2 + p3;
                    ss += wspr_pr3[i] ? cmet : -cmet;
                }
                ss = ss / totp;
                if (ss > syncmax) {
                    syncmax = ss;
                    best_shift = lag;
                    fbest = f0;
                }
            }
        }
        Py_END_ALLOW_THREADS;

        Py_DECREF(c_a);
        return Py_BuildValue("ddl", syncmax, fbest, best_shift);
    }
}

/* wspr_ncsd(c, f1, shift1, drift1, symfac, nblock, bitmetric) -> symbols
 *
 * Noncoherent sequence detection (block demodulation) of wsprd.c:
 * 162 soft symbols as uint8 (128 + clipped soft value).
 */
static PyObject *
wspr_ncsd(PyObject *self, PyObject *args)
{
    PyObject *c_o;
    double f1, drift1;
    long shift1, symfac, nblock, bitmetric;
    PyArrayObject *c_a, *sym_a;
    npy_intp dims[1] = {WSPR_NSYM};

    if (!PyArg_ParseTuple(args, "Odldlll", &c_o, &f1, &shift1, &drift1,
                          &symfac, &nblock, &bitmetric))
        return NULL;
    c_a = as_array(c_o, NPY_CDOUBLE, 1, "c");
    if (!c_a)
        return NULL;
    sym_a = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_UINT8, 0);
    if (!sym_a) {
        Py_DECREF(c_a);
        return NULL;
    }

    {
        const double *cd = (const double *)PyArray_DATA(c_a);
        const npy_intp npts = PyArray_DIM(c_a, 0);
        uint8_t *symbols = (uint8_t *)PyArray_DATA(sym_a);
        double zr[4][WSPR_NSYM], zi[4][WSPR_NSYM];
        double cf[4][WSPR_NSYM], sf[4][WSPR_NSYM];
        double p[8], fsymb[WSPR_NSYM];
        double fsum = 0.0, f2sum = 0.0, fac;
        long nseq = 1L << nblock;
        int i, j, ib, b, itone, imask;

        Py_BEGIN_ALLOW_THREADS;
        wspr_tone_corr(cd, npts, f1, drift1, shift1, zr, zi, cf, sf);
        for (i = 0; i < WSPR_NSYM; i += nblock) {
            for (j = 0; j < nseq; j++) {
                double xi = 0.0, xq = 0.0, cm = 1.0, sm = 0.0;
                for (ib = 0; ib < nblock; ib++) {
                    double cmp, smp;
                    b = (j >> (nblock - 1 - ib)) & 1;
                    itone = wspr_pr3[i + ib] + 2 * b;
                    xi += zr[itone][i + ib] * cm + zi[itone][i + ib] * sm;
                    xq += zi[itone][i + ib] * cm - zr[itone][i + ib] * sm;
                    cmp = cf[itone][i + ib] * cm - sf[itone][i + ib] * sm;
                    smp = sf[itone][i + ib] * cm + cf[itone][i + ib] * sm;
                    cm = cmp;
                    sm = smp;
                }
                p[j] = sqrt(xi * xi + xq * xq);
            }
            for (ib = 0; ib < nblock; ib++) {
                double xm1 = 0.0, xm0 = 0.0;
                imask = 1 << (nblock - 1 - ib);
                for (j = 0; j < nseq; j++) {
                    if ((j & imask) != 0 && p[j] > xm1)
                        xm1 = p[j];
                    if ((j & imask) == 0 && p[j] > xm0)
                        xm0 = p[j];
                }
                fsymb[i + ib] = xm1 - xm0;
                if (bitmetric)
                    fsymb[i + ib] /= (xm1 > xm0 ? xm1 : xm0);
            }
        }
        for (i = 0; i < WSPR_NSYM; i++) {
            fsum += fsymb[i] / 162.0;
            f2sum += fsymb[i] * fsymb[i] / 162.0;
        }
        fac = sqrt(f2sum - fsum * fsum);
        if (!(fac > 0.0))
            fac = 1.0;
        for (i = 0; i < WSPR_NSYM; i++) {
            double v = symfac * fsymb[i] / fac;
            if (v > 127.0)
                v = 127.0;
            if (v < -128.0)
                v = -128.0;
            symbols[i] = (uint8_t)(v + 128.0);
        }
        Py_END_ALLOW_THREADS;
    }
    Py_DECREF(c_a);
    return (PyObject *)sym_a;
}

static int
wspr_par32(uint32_t v)
{
    v ^= v >> 16;
    v ^= v >> 8;
    v ^= v >> 4;
    return (0x6996 >> (v & 0xf)) & 1;
}

/* branch symbol pair for an encoder state (ENCODE macro in fano.h) */
#define WSPR_ENCODE(state) \
    ((wspr_par32((uint32_t)(state) & WSPR_POLY1) << 1) | \
     wspr_par32((uint32_t)(state) & WSPR_POLY2))

struct wspr_node {
    uint64_t encstate; /* encoder state of next node */
    long gamma;        /* cumulative metric to this node */
    int metrics[4];    /* metrics indexed by all possible tx symbols */
    int tm[2];         /* sorted metrics for current hypotheses */
    int i;             /* current branch being tested */
};

/* wspr_fano(symbols, mettab, delta, maxcycles) -> (ok, data, metric, cycles)
 *
 * Fano sequential decoder for the WSPR K=32 r=1/2 code; transcription of
 * fano() in fano.c with nbits = 81.  symbols are the 162 deinterleaved
 * soft symbols; mettab is the int32 (2, 256) metric table.  ok is 0 when
 * the decoder timed out, data holds the 11 decoded bytes (50 bits + tail).
 */
static PyObject *
wspr_fano(PyObject *self, PyObject *args)
{
    PyObject *sym_o, *met_o;
    long delta, maxcycles;
    PyArrayObject *sym_a = NULL, *met_a = NULL;
    const unsigned nbits = 81;
    uint8_t data[11];
    long metric;
    unsigned long cycles;
    int ok;

    if (!PyArg_ParseTuple(args, "OOll", &sym_o, &met_o, &delta, &maxcycles))
        return NULL;
    sym_a = as_array(sym_o, NPY_UINT8, 1, "symbols");
    met_a = as_array(met_o, NPY_INT32, 2, "mettab");
    if (!sym_a || !met_a)
        goto fail;
    if (PyArray_DIM(sym_a, 0) != WSPR_NSYM || PyArray_DIM(met_a, 0) != 2 ||
        PyArray_DIM(met_a, 1) != 256) {
        PyErr_SetString(PyExc_ValueError, "wspr_fano: bad array shape");
        goto fail;
    }

    {
        const uint8_t *symbols = (const uint8_t *)PyArray_DATA(sym_a);
        const int32_t *m0 = (const int32_t *)PyArray_DATA(met_a);
        const int32_t *m1 = m0 + 256;
        struct wspr_node nodes[82];
        struct wspr_node *np_ = nodes;
        struct wspr_node *lastnode = &nodes[nbits - 1];
        struct wspr_node *tail = &nodes[nbits - 31];
        unsigned long maxtotal = (unsigned long)maxcycles * nbits;
        unsigned long i;
        long t, ngamma;
        int mm0, mm1;
        unsigned lsym, b;

        Py_BEGIN_ALLOW_THREADS;
        for (np_ = nodes; np_ <= lastnode; np_++) {
            int s0 = symbols[0], s1 = symbols[1];
            np_->metrics[0] = m0[s0] + m0[s1];
            np_->metrics[1] = m0[s0] + m1[s1];
            np_->metrics[2] = m1[s0] + m0[s1];
            np_->metrics[3] = m1[s0] + m1[s1];
            symbols += 2;
        }
        np_ = nodes;
        np_->encstate = 0;

        lsym = WSPR_ENCODE(np_->encstate);
        mm0 = np_->metrics[lsym];
        mm1 = np_->metrics[3 ^ lsym];
        if (mm0 > mm1) {
            np_->tm[0] = mm0;
            np_->tm[1] = mm1;
        } else {
            np_->tm[0] = mm1;
            np_->tm[1] = mm0;
            np_->encstate++;
        }
        np_->i = 0;
        np_->gamma = t = 0;

        for (i = 1; i <= maxtotal; i++) {
            ngamma = np_->gamma + np_->tm[np_->i];
            if (ngamma >= t) {
                if (np_->gamma < t + delta) {
                    while (ngamma >= t + delta)
                        t += delta;
                }
                np_[1].gamma = ngamma;
                np_[1].encstate = np_->encstate << 1;
                if (++np_ == (lastnode + 1))
                    break;
                lsym = WSPR_ENCODE(np_->encstate);
                if (np_ >= tail) {
                    np_->tm[0] = np_->metrics[lsym];
                } else {
                    mm0 = np_->metrics[lsym];
                    mm1 = np_->metrics[3 ^ lsym];
                    if (mm0 > mm1) {
                        np_->tm[0] = mm0;
                        np_->tm[1] = mm1;
                    } else {
                        np_->tm[0] = mm1;
                        np_->tm[1] = mm0;
                        np_->encstate++;
                    }
                }
                np_->i = 0;
                continue;
            }
            for (;;) {
                if (np_ == nodes || np_[-1].gamma < t) {
                    t -= delta;
                    if (np_->i != 0) {
                        np_->i = 0;
                        np_->encstate ^= 1;
                    }
                    break;
                }
                if (--np_ < tail && np_->i != 1) {
                    np_->i++;
                    np_->encstate ^= 1;
                    break;
                }
            }
        }
        metric = np_->gamma;
        memset(data, 0, sizeof(data));
        for (b = 0; b < (nbits >> 3); b++)
            data[b] = (uint8_t)nodes[7 + 8 * b].encstate;
        cycles = i + 1;
        ok = (i >= maxtotal) ? 0 : 1;
        Py_END_ALLOW_THREADS;
    }

    Py_DECREF(sym_a);
    Py_DECREF(met_a);
    return Py_BuildValue("iy#lk", ok, (const char *)data, (Py_ssize_t)11,
                         metric, cycles);

fail:
    Py_XDECREF(sym_a);
    Py_XDECREF(met_a);
    return NULL;
}

static PyMethodDef ckernel_methods[] = {
    {"bp_hybrid", bp_hybrid, METH_VARARGS,
     "Belief-propagation loop for the (174,91) code."},
    {"osd_ge", osd_ge, METH_VARARGS,
     "In-place Gaussian elimination for OSD."},
    {"osd", osd, METH_VARARGS,
     "Ordered-statistics decode of one LLR vector."},
    {"sync8d", sync8d, METH_VARARGS,
     "Sync power of a downsampled FT8 signal."},
    {"ft4_sync_search", ft4_sync_search, METH_VARARGS,
     "Grid search of FT4 sync power over frequency/time offsets."},
    {"crc14_check", crc14_check, METH_VARARGS,
     "Check a 91-bit message+CRC block."},
    {"wspr_sync_demod", wspr_sync_demod, METH_VARARGS,
     "WSPR sync search over lag/frequency grids."},
    {"wspr_ncsd", wspr_ncsd, METH_VARARGS,
     "WSPR noncoherent block demodulation to 162 soft symbols."},
    {"wspr_fano", wspr_fano, METH_VARARGS,
     "Fano sequential decoder for the WSPR K=32 convolutional code."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef ckernels_module = {
    PyModuleDef_HEAD_INIT, "_ckernels",
    "Compiled kernels for the FT8/FT4 decoder hot paths.", -1,
    ckernel_methods,
};

PyMODINIT_FUNC
PyInit__ckernels(void)
{
    import_array();
    return PyModule_Create(&ckernels_module);
}
