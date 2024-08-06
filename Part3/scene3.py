from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene3_3(VoiceoverScene):
    def construct(self):

        self.wait(2)

        # Quality of reconstruction with different latent dimensions
        self.next_section(skip_animations=False)

        encoder = Polygon(
            [-1, 1, 0], [1, 0.4, 0], [1, -0.4, 0], [-1, -1, 0], color=PURPLE
        )
        bottleneck = Rectangle(width=1.5, height=0.8, color=BLUE)
        bottleneck.next_to(encoder, direction=RIGHT, buff=0.4)
        decoder = Polygon(
            [-1, 0.4, 0], [1, 1, 0], [1, -1, 0], [-1, -0.4, 0], color=PURPLE
        ).next_to(bottleneck, direction=RIGHT, buff=0.4)

        autoencoder = (
            VGroup(
                encoder,
                bottleneck,
                decoder,
            )
            .scale(0.9)
            .move_to(ORIGIN)
        )

        input_img = (
            ImageMobject("images/instances1D/image_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        input_rect = SurroundingRectangle(
            input_img, color=WHITE, stroke_width=2, buff=0.0, z_index=1
        )
        input_img = Group(input_img, input_rect).next_to(
            autoencoder, direction=LEFT, buff=1.0
        )
        arrowin = Arrow(
            input_img.get_right(), autoencoder.get_left(), color=WHITE, buff=0.2
        )
        output_img = (
            ImageMobject("images/instances1D/recon_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
        )
        output_rect = SurroundingRectangle(
            output_img, color=WHITE, stroke_width=2, buff=0.0, z_index=1
        )
        output_img = Group(output_img, output_rect).next_to(
            autoencoder, direction=RIGHT, buff=1.0
        )
        arrowout = Arrow(
            autoencoder.get_right(), output_img.get_left(), color=WHITE, buff=0.2
        )

        title = Tex("Latent dimension = 1").scale(1.5).to_edge(UP)

        self.play(FadeIn(autoencoder))
        self.play(FadeIn(title))

        self.wait(0.7)

        self.play(
            FadeIn(
                input_img,
                arrowin,
            )
        )

        self.play(FadeIn(output_img, arrowout))

        input_img_1 = (
            ImageMobject("images/instances1D/image_1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
            .move_to(input_img)
        )

        output_img_1 = (
            ImageMobject("images/instances1D/recon_1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
            .move_to(output_img)
        )

        self.play(
            FadeOut(input_img[0]),
            FadeIn(input_img_1),
        )
        self.play(
            FadeOut(output_img[0]),
            FadeIn(output_img_1),
        )

        input_img_2 = (
            ImageMobject("images/instances1D/image_2.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
            .move_to(input_img)
        )

        output_img_2 = (
            ImageMobject("images/instances1D/recon_2.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
            .move_to(output_img)
        )

        self.play(
            FadeOut(input_img_1),
            FadeIn(input_img_2),
        )
        self.play(
            FadeOut(output_img_1),
            FadeIn(output_img_2),
        )

        self.wait(0.5)

        input_img_3 = (
            ImageMobject("images/instances1D/image_3.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
            .move_to(input_img)
        )

        output_img_3 = (
            ImageMobject("images/instances1D/recon_3.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
            .move_to(output_img)
        )

        self.play(FadeOut(input_img_2), FadeIn(input_img_3))
        self.play(FadeOut(output_img_2), FadeIn(output_img_3))

        input_img_4 = (
            ImageMobject("images/instances1D/image_4.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
            .move_to(input_img)
        )

        output_img_4 = (
            ImageMobject("images/instances1D/recon_4.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
            .move_to(output_img)
        )

        self.play(
            FadeOut(input_img_3),
            FadeIn(input_img_4),
        )
        self.play(
            FadeOut(output_img_3),
            FadeIn(output_img_4),
        )

        self.wait(0.5)

        self.play(FadeOut(input_img_4, output_img_4))

        # 2D latent reconstruction

        title2 = Tex("Latent dimension = 2").scale(1.5).to_edge(UP)
        self.play(Transform(title, title2))
        self.play(ShowPassingFlash(Underline(title)))

        input_img = (
            ImageMobject("images/instances2D/image_6.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
            .move_to(input_img_4)
        )

        output_img = (
            ImageMobject("images/instances2D/recon_6.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
            .move_to(output_img_4)
        )

        self.play(FadeIn(input_img))
        self.play(FadeIn(output_img))

        self.wait(0.6)

        # 9D latent reconstruction

        self.play(FadeOut(input_img, output_img))

        title3 = Tex("Latent dimension = 9").scale(1.5).to_edge(UP)
        self.play(Transform(title, title3))
        self.play(ShowPassingFlash(Underline(title)))
        input_img_1 = (
            ImageMobject("images/instances9D/image_6.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
            .move_to(input_img)
        )

        output_img_1 = (
            ImageMobject("images/instances9D/recon_6.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(10)
            .move_to(output_img)
        )

        self.play(FadeIn(input_img_1))
        self.play(FadeIn(output_img_1))

        self.wait(0.6)

        self.play(
            FadeOut(
                input_img_1,
                output_img_1,
                autoencoder,
                title,
                arrowin,
                arrowout,
                input_rect,
                output_rect,
            )
        )
        txt = Tex("Why do we care about reconstruction quality ?")
        self.play(Write(txt))
        self.wait(2)

        self.play(FadeOut(txt))
        txt = Tex("Reconstruction quality = Latent space quality")
        self.play(Write(txt))
        self.wait(2)
        self.play(FadeOut(txt))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene3_3()
    scene.render()
