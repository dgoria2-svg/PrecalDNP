// app/src/main/java/com/dg/precaldnp/vision/PupilFrameEngine3250.kt
package com.dg.precaldnp.vision

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import androidx.annotation.RawRes
import com.dg.precaldnp.R
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfRect
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import kotlin.math.hypot
import kotlin.math.min

/**
 * Engine 1:
 * - Detecta cara y pupilas.
 * - Valida encuadre (márgenes laterales y altura de pupilas).
 * - NO usa escala ni FIL. Es puramente geométrico en píxeles.
 *
 * Pensado para ser usado desde DnpFaceCaptureActivity antes de cualquier otra cosa.
 */
class PupilFrameEngine3250(private val context: Context) {

    companion object {
        private const val TAG = "PupilFrameEngine3250"

        // Márgenes mínimos laterales (15% de ancho de la foto a cada lado).
        private const val MIN_SIDE_MARGIN_FRACTION = 0.15f

        // Rango razonable para la altura normalizada de las pupilas (0 = arriba, 1 = abajo).
        // El usuario pidió: "60% desde abajo" ~ 40% desde arriba, alrededor de la mitad superior.
        // Le damos un rango generoso: [0.25, 0.75]
        private const val MIN_PUPIL_Y_NORM = 0.25f
        private const val MAX_PUPIL_Y_NORM = 0.75f
    }

    // --------- Tipos de respuesta ---------

    data class Pupils3250(
        val left: Point?,   // ojo izquierdo en coordenadas de imagen (x pequeña)
        val right: Point?   // ojo derecho en coordenadas de imagen (x grande)
    )

    data class FrameCheck3250(
        val ok: Boolean,
        val reason: String? = null,
        val faceRect: Rect? = null,
        val pupils: Pupils3250? = null,
        val midlineX: Float? = null
    )

    // --------- Cascadas Haar ---------

    private val faceCascade by lazy {
        loadCascade(
            R.raw.haarcascade_frontalface_default,
            "haarcascade_frontalface_default.xml"
        )
    }

    private val eyeCascade by lazy {
        loadCascade(
            R.raw.haarcascade_eye,
            "haarcascade_eye.xml"
        )
    }

    private val eyeCascadeGlasses by lazy {
        loadCascade(
            R.raw.haarcascade_eye_tree_eyeglasses,
            "haarcascade_eye_tree_eyeglasses.xml"
        )
    }

    private fun loadCascade(@RawRes resId: Int, name: String): CascadeClassifier {
        val outFile = java.io.File(context.cacheDir, name)
        try {
            context.resources.openRawResource(resId).use { input ->
                java.io.FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Error copiando cascade $name", t)
        }
        val cascade = CascadeClassifier(outFile.absolutePath)
        if (cascade.empty()) {
            Log.e(TAG, "No se pudo cargar $name")
        }
        return cascade
    }

    // ================================================================
    //  API PÚBLICA
    // ================================================================

    /**
     * Detecta cara+pupilas y valida:
     * - Márgenes laterales mínimos (15% a cada lado).
     * - Pupilas en banda vertical [25%, 75%] de la altura de la foto.
     *
     * @param bitmap imagen YA orientada (como la que generás desde Exif).
     */
    fun detectAndCheckFrame(bitmap: Bitmap): FrameCheck3250 {
        val matRgba = Mat()
        val matGray = Mat()
        return try {
            Utils.bitmapToMat(bitmap, matRgba)
            Imgproc.cvtColor(matRgba, matGray, Imgproc.COLOR_RGBA2GRAY)
            Imgproc.equalizeHist(matGray, matGray)
            Imgproc.GaussianBlur(matGray, matGray, Size(3.0, 3.0), 0.0)

            val imgW = matGray.width()
            val imgH = matGray.height()

            // 1) Cara (si no hay cara, no seguimos).
            val face = detectFace(matGray) ?: return FrameCheck3250(
                ok = false,
                reason = "No se detectó la cara. Acercá un poco o mejorá la luz."
            )

            // 2) Pupilas seed (en píxeles).
            val pupils = detectPupils(matGray, face)
            if (pupils.left == null || pupils.right == null) {
                return FrameCheck3250(
                    ok = false,
                    reason = "No se pudieron detectar ambas pupilas."
                )
            }

            // 3) Chequeos de encuadre: márgenes y altura.
            val marginsOk = checkSideMargins(face, imgW)
            if (!marginsOk) {
                return FrameCheck3250(
                    ok = false,
                    reason = "La cabeza está demasiado pegada a los bordes. Dejá ~15% de margen."
                )
            }

            val pupilsYOk = checkPupilHeight(pupils, imgH)
            if (!pupilsYOk) {
                return FrameCheck3250(
                    ok = false,
                    reason = "Las pupilas están demasiado arriba o abajo. Reencuadrá la cara."
                )
            }

            val midlineX = computeMidlineX(pupils, imgW)

            FrameCheck3250(
                ok = true,
                reason = null,
                faceRect = face,
                pupils = pupils,
                midlineX = midlineX
            )
        } catch (t: Throwable) {
            Log.e(TAG, "Error en detectAndCheckFrame", t)
            FrameCheck3250(
                ok = false,
                reason = "Error interno en motor de pupilas.",
                faceRect = null,
                pupils = null,
                midlineX = null
            )
        } finally {
            matRgba.release()
            matGray.release()
        }
    }

    // ================================================================
    //  Cara y pupilas
    // ================================================================

    private fun detectFace(gray: Mat): Rect? {
        val faces = MatOfRect()
        try {
            faceCascade.detectMultiScale(
                gray,
                faces,
                1.1,
                3,
                0,
                Size(80.0, 80.0),
                Size()
            )
            val arr = faces.toArray()
            return arr.maxByOrNull { it.width * it.height }
        } finally {
            faces.release()
        }
    }

    private fun detectPupils(gray: Mat, face: Rect): Pupils3250 {
        // ROI donde esperamos los ojos dentro de la cara
        val roiEyes = Rect(
            face.x,
            face.y + (face.height * 0.2).toInt(),
            face.width,
            (face.height * 0.6).toInt()
        )

        val sub = gray.submat(roiEyes)

        fun detectEyesWith(cascade: CascadeClassifier, mat: Mat): Array<Rect> {
            val eyes = MatOfRect()
            try {
                cascade.detectMultiScale(
                    mat,
                    eyes,
                    1.1,
                    2,
                    0,
                    Size(24.0, 24.0),
                    Size()
                )
                return eyes.toArray()
            } finally {
                eyes.release()
            }
        }

        var candidates = detectEyesWith(eyeCascade, sub).toList()
        if (candidates.size < 2) {
            val withGlasses = detectEyesWith(eyeCascadeGlasses, sub)
            val merged = ArrayList<Rect>(candidates)
            for (r in withGlasses) {
                val cNew = Point(
                    (roiEyes.x + r.x + r.width / 2).toDouble(),
                    (roiEyes.y + r.y + r.height / 2).toDouble()
                )
                val tooClose = merged.any { old ->
                    val cOld = Point(
                        (roiEyes.x + old.x + old.width / 2).toDouble(),
                        (roiEyes.y + old.y + old.height / 2).toDouble()
                    )
                    val d = hypot(cOld.x - cNew.x, cOld.y - cNew.y)
                    d < min(old.width, old.height) * 0.5
                }
                if (!tooClose) merged.add(r)
            }
            candidates = merged
        }

        val found = mutableListOf<Point>()
        for (e in candidates.sortedBy { it.x }.take(2)) {
            val eyeRect = Rect(
                roiEyes.x + e.x,
                roiEyes.y + e.y,
                e.width,
                e.height
            )
            val eyeMat = gray.submat(eyeRect)
            val center = detectDarkCircleCenter(eyeMat)
            eyeMat.release()
            if (center != null) {
                found += Point(
                    eyeRect.x + center.x,
                    eyeRect.y + center.y
                )
            }
        }

        sub.release()

        if (found.isEmpty()) {
            return Pupils3250(null, null)
        }

        // Ojo izquierdo = x más chico; derecho = x más grande
        val left = found.minByOrNull { it.x }
        val right = if (found.size >= 2) found.maxByOrNull { it.x } else null

        return Pupils3250(left, right)
    }

    private fun detectDarkCircleCenter(eyeGray: Mat): Point? {
        val g = Mat()
        val thr = Mat()
        val circles = Mat()
        return try {
            Imgproc.equalizeHist(eyeGray, g)
            Imgproc.GaussianBlur(g, g, Size(3.0, 3.0), 0.0)
            Imgproc.threshold(
                g,
                thr,
                0.0,
                255.0,
                Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU
            )

            Imgproc.HoughCircles(
                g,
                circles,
                Imgproc.HOUGH_GRADIENT,
                1.0,
                g.rows() / 8.0,
                120.0,
                15.0,
                3,
                g.rows() / 3
            )

            if (!circles.empty() && circles.cols() > 0) {
                val data = circles.get(0, 0)
                Point(data[0], data[1])
            } else {
                // Fallback por contornos
                val contours = mutableListOf<MatOfPoint>()
                val hierarchy = Mat()
                try {
                    Imgproc.findContours(
                        thr,
                        contours,
                        hierarchy,
                        Imgproc.RETR_EXTERNAL,
                        Imgproc.CHAIN_APPROX_SIMPLE
                    )
                    val best = contours.maxByOrNull { Imgproc.contourArea(it) }
                    best?.let {
                        val m = Imgproc.moments(it)
                        if (m.m00 != 0.0) {
                            Point(m.m10 / m.m00, m.m01 / m.m00)
                        } else null
                    }
                } finally {
                    hierarchy.release()
                    contours.forEach { it.release() }
                }
            }
        } finally {
            g.release()
            thr.release()
            circles.release()
        }
    }

    private fun checkSideMargins(face: Rect, imgW: Int): Boolean {
        val leftMarginPx = face.x
        val rightMarginPx = imgW - (face.x + face.width)
        val leftFrac = leftMarginPx.toFloat() / imgW.toFloat()
        val rightFrac = rightMarginPx.toFloat() / imgW.toFloat()
        val minFrac = min(leftFrac, rightFrac)
        Log.d(TAG, "Márgenes laterales: left=$leftFrac right=$rightFrac")
        return minFrac >= MIN_SIDE_MARGIN_FRACTION
    }

    private fun checkPupilHeight(pupils: Pupils3250, imgH: Int): Boolean {
        val pL = pupils.left ?: return false
        val pR = pupils.right ?: return false
        val yAvg = ((pL.y + pR.y) / 2.0).toFloat()
        val yNorm = yAvg / imgH.toFloat()
        Log.d(TAG, "Altura pupilas: yNorm=$yNorm")
        return yNorm in MIN_PUPIL_Y_NORM..MAX_PUPIL_Y_NORM
    }

    private fun computeMidlineX(p: Pupils3250, w: Int): Float {
        val left = p.left
        val right = p.right
        return if (left != null && right != null) {
            ((left.x + right.x) / 2.0).toFloat()
        } else {
            w / 2f
        }
    }
}
