/* C kernels for the decoder hot paths.
 *
 * Direct transcriptions of the WSJT-X Fortran inner loops
 * (bpdecode174_91.f90, osd174_91.f90, sync8d.f90) exposed to Python.
 * The pure-numpy implementations in ldpc.py / decode.py remain as the
 * fallback when this module is unavailable.
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
