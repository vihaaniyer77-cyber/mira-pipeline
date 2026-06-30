from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import os

def create_presentation(output_path):
    c = canvas.Canvas(output_path, pagesize=landscape(letter))
    width, height = landscape(letter)

    def draw_slide(title, bullets, image_path=None):
        c.setFont("Helvetica-Bold", 24)
        c.setFillColor(colors.darkblue)
        c.drawString(50, height - 60, title)
        
        c.setStrokeColor(colors.black)
        c.line(50, height - 70, width - 50, height - 70)
        
        c.setFont("Helvetica", 18)
        c.setFillColor(colors.black)
        y = height - 120
        
        # If there's an image, limit text width
        text_width_limit = width / 2 if image_path else width - 100
        
        for bullet in bullets:
            # Simple text wrapping logic
            words = bullet.split()
            line = "- "
            for word in words:
                if c.stringWidth(line + word + " ", "Helvetica", 18) < text_width_limit:
                    line += word + " "
                else:
                    c.drawString(50, y, line)
                    y -= 30
                    line = "  " + word + " "
            c.drawString(50, y, line)
            y -= 40
            
        if image_path and os.path.exists(image_path):
            # Draw image on the right
            c.drawImage(image_path, width/2 + 20, height - 400, width=320, height=320, preserveAspectRatio=True)
            
        c.showPage()

    # SLIDE 1
    c.setFont("Helvetica-Bold", 36)
    c.setFillColor(colors.darkblue)
    c.drawCentredString(width/2, height/2 + 40, "MIRA Transient Pipeline")
    c.setFont("Helvetica", 24)
    c.setFillColor(colors.black)
    c.drawCentredString(width/2, height/2, "Real-Time Offline Discovery & Autonomous Follow-Up")
    c.setFont("Helvetica-Oblique", 18)
    c.drawCentredString(width/2, height/2 - 40, "My comprehensive architecture, implementation, and verification")
    c.showPage()

    # SLIDE 2
    draw_slide("My Project Objectives", [
        "I designed this pipeline to control a 36-inch telescope system.",
        "A critical requirement I solved is ensuring a fully air-gapped, offline environment.",
        "I need the system to detect transients and slew automatically with zero human input.",
        "I am building an end-to-end framework: from raw FITS ingestion to final alerts."
    ])

    # SLIDE 3
    draw_slide("The Computational Challenge", [
        "Transient detection is hard: we often have NO historical reference templates.",
        "Atmospheric blurring (seeing) constantly changes the shape of stars.",
        "Cosmic rays and hot pixels create massive numbers of false positives.",
        "I had to build a robust mathematical system to conquer these constraints."
    ])

    # SLIDE 4
    draw_slide("My Dual-Engine Architecture", [
        "I designed a parallel Dual-Engine approach to catch every type of event.",
        "Engine A: Optimal Image Subtraction (OIS) - I built this to hunt for brand-new objects.",
        "Engine B: Rolling Z-Score Photometry - I built this to monitor known catalog objects.",
        "These engines run simultaneously to cross-verify anomalies."
    ])

    # SLIDE 5
    draw_slide("Dynamic Reference Generation", [
        "Because historical templates often don't exist, I built a Burn-In Phase.",
        "The telescope captures 5 frames of the new target and median-stacks them.",
        "This median combination dynamically erases fast transients and cosmic rays.",
        "This creates a pristine Nightly Reference Template entirely on the fly!"
    ])

    # SLIDE 5b
    draw_slide("Engine A: Optimal Image Subtraction", [
        "Even with a nightly template, atmospheric blurring changes frame-to-frame.",
        "To solve this, I completely rewrote the subtraction engine.",
        "I implemented a dynamic Least-Squares kernel fitting algorithm.",
        "Now, my engine calculates the exact atmospheric blur to match images perfectly."
    ])

    # SLIDE 6
    draw_slide("Engine B: Z-Score Photometry", [
        "For my second engine, I wrote a rolling-window statistical model.",
        "I perform rapid forced aperture photometry on all known coordinates.",
        "I maintain an in-memory 5-frame history for every object.",
        "If a star breaks the 3.5-sigma threshold, I immediately generate a slew alert."
    ])

    # SLIDE 7
    draw_slide("Solving the Photometric Noise Floor", [
        "I discovered a critical mathematical edge-case during my testing.",
        "If a star was perfectly stable, its standard deviation dropped to zero.",
        "This artificially tiny variance caused massive false-positive Z-score spikes.",
        "I fixed this by engineering a custom minimum-noise floor (epsilon guard) into the math."
    ])

    # SLIDE 8
    draw_slide("Spatial Profile Vetting", [
        "I wrote strict spatial filters to instantly discard cosmic rays.",
        "I compute the Full Width at Half Maximum (FWHM) of every detection.",
        "I also compute the ellipticity to reject elongated subtraction artifacts (dipoles).",
        "I corrected standard astronomy formulas to accurately measure 2.3548-sigma."
    ])

    # SLIDE 9
    draw_slide("Temporal Verification", [
        "A single frame anomaly is not enough to trigger my telescope.",
        "I built a TemporalVerifier class to track object persistence.",
        "I require an anomaly to persist across consecutive frames.",
        "This drastically cuts down on random noise and fleeting artifacts."
    ])

    # SLIDE 10
    draw_slide("My Simulation Environment", [
        "Since I don't have live sky data yet, I built a full simulation engine.",
        "I inject background noise, Poisson statistics, and variable seeing.",
        "I artificially blurred the target frames to prove my dynamic kernel fitting works.",
        "I simulated 4 profiles: No Variability, Pulsator, Stellar Flare, and Supernova."
    ])

    # SLIDE 11
    draw_slide("Result: Stable Stars Subtracted Cleanly", [
        "Because of my dynamic Least-Squares kernel, stable stars vanish.",
        "The difference image shows pure background noise.",
        "My noise floor fix keeps the Z-score perfectly flat, ignoring the star."
    ], "difference_image_No_Variability.png")

    # SLIDE 12
    draw_slide("Result: Detecting Pulsators via Variance", [
        "Pulsators change brightness cyclically, meaning they often evade Z-score spikes.",
        "I added a Variance Analysis trigger to the Photometry Engine to catch them.",
        "When the rolling variance exceeded 3x the baseline noise, my pipeline flagged it.",
        "The system now successfully categorizes this as a Variable Star alert."
    ], "variance_Pulsator.png")

    # SLIDE 13
    draw_slide("Result: Instant Stellar Flare Detection", [
        "I simulated a sudden stellar flare peaking at Frame 10.",
        "My Z-score engine instantly calculated an astronomical variance spike.",
        "The alert threshold was breached instantly.",
        "My pipeline correctly generated an immediate alert."
    ], "z_score_Stellar_Flare.png")

    # SLIDE 14
    draw_slide("Result: Clean Supernova Extraction", [
        "I simulated a slowly rising supernova in an empty patch of sky.",
        "Because it wasn't in my reference frame, it passed through my subtraction engine.",
        "My Least-Squares kernel perfectly erased the background stars.",
        "Only the Supernova remained, isolated perfectly in my difference image."
    ], "difference_image_Supernova.png")

    # SLIDE 15
    draw_slide("Automated Test Suite Design", [
        "I wrote a comprehensive suite of unit tests using pytest.",
        "I achieved full coverage across my calibration, subtraction, and photometry modules.",
        "I discovered and fixed Numpy structured array crashes during my testing.",
        "I ensure my mathematical formulas are verified automatically on every run."
    ])

    # SLIDE 16
    draw_slide("My Test Results", [
        "I successfully executed 8 out of 8 core module tests.",
        "I eliminated all hidden RuntimeWarnings and zero-division errors.",
        "My code is stable, mathematically verified, and ready for deployment.",
        "All tests pass perfectly in the air-gapped virtual environment."
    ])

    # SLIDE 17
    draw_slide("Integration: Real-World Coordinates", [
        "My pipeline isn't just pixel math; I integrated Astropy WCS.",
        "When an anomaly is verified, I extract the exact Right Ascension and Declination.",
        "This translates the raw pixel data into actionable astronomical coordinates.",
        "These coordinates are formatted and ready for the telescope mount."
    ])

    # SLIDE 18
    draw_slide("Next Phase: Hardware Automation", [
        "Now that my algorithms are perfect, I am ready for Phase 4.",
        "I will integrate PyINDI and PyASCOM to take direct control of the mount.",
        "My pipeline will automatically pass the calculated RA/Dec to the motors.",
        "This will achieve my ultimate goal of completely autonomous follow-up."
    ])

    # SLIDE 19
    draw_slide("Conclusion", [
        "I successfully developed a Real-Time Offline Transient Pipeline.",
        "I solved critical challenges with Optimal Image Subtraction and false positives.",
        "My dual-engine architecture is robust, mathematically verified, and tested.",
        "I am fully prepared to deploy this on the 36-inch telescope system."
    ])

    c.save()

if __name__ == "__main__":
    create_presentation("/Users/vihaa/Desktop/transient_pipeline_code/MIRA_Pipeline_Presentation.pdf")
