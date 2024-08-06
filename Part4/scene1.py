from manim import *
from manim_voiceover import VoiceoverScene
import numpy as np
import matplotlib.pyplot as plt


class Scene4_1(VoiceoverScene):
    def construct(self):
        self.wait(2)

        # Intro
        self.next_section(skip_animations=False)

        imgs = Group(
            ImageMobject("images/digits/image_0.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(0.1),
            ImageMobject("images/digits/image_1.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(0.1),
            ImageMobject("images/digits/image_2.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(0.1),
            ImageMobject("images/digits/image_3.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(0.1),
            ImageMobject("images/digits/image_4.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(0.1),
        )
        self.add(*imgs)

        for img in imgs:
            img.add(SurroundingRectangle(img, color=WHITE, stroke_width=2, buff=0))

        target_positions = [
            3 * LEFT + 2 * UP,
            3 * RIGHT + 2 * UP,
            3 * LEFT + 2 * DOWN,
            3 * RIGHT + 2 * DOWN,
            ORIGIN,
        ]

        self.wait(0.2)
        for img, target_position in zip(imgs, target_positions):
            self.play(img.animate.move_to(target_position).scale(10 / 0.1))

        self.wait(1)

        hospital = SVGMobject(
            "svg/hospital.svg",
            opacity=1,
            fill_opacity=1,
            stroke_color=WHITE,
            fill_color=WHITE,
        ).scale(1.5)

        patient = (
            SVGMobject(
                "svg/patient-sitting.svg",
                opacity=1,
                fill_opacity=1,
                stroke_color=WHITE,
                fill_color=WHITE,
            )
            .scale(1.5)
            .shift(2 * RIGHT)
        )

        self.play(FadeOut(imgs))
        self.play(FadeIn(hospital))
        self.wait(1)
        self.play(hospital.animate.shift(3 * LEFT))
        self.play(FadeIn(patient))

        self.wait(0.8)

        self.play(FadeOut(hospital))
        img = (
            ImageMobject(f"images/brain1/slice0.png")
            .scale(4)
            .rotate(180 * DEGREES)
            .shift(3 * LEFT)
        )
        self.add(img)

        mri_txt = Tex("MRI Scan").next_to(img, UP)
        self.play(FadeIn(mri_txt))

        for i in range(1, 96):
            img2 = (
                ImageMobject(f"images/brain1/slice{i}.png")
                .scale(4)
                .rotate(180 * DEGREES)
                .shift(3 * LEFT)
            )

            self.remove(img)
            self.add(img2)
            img = img2
            self.wait(0.05)

        self.wait(0.5)

        note = SVGMobject(
            "svg/patient-note.svg",
            opacity=1,
            fill_opacity=1,
            stroke_color=WHITE,
            fill_color=WHITE,
        ).shift(DOWN)

        report = (
            SVGMobject(
                "svg/report.svg",
                opacity=1,
                fill_opacity=1,
                stroke_color=WHITE,
                fill_color=WHITE,
            )
        ).next_to(note, RIGHT)

        VGroup(note, report).move_to(ORIGIN)

        self.play(FadeOut(img, mri_txt))
        self.play(FadeOut(patient))
        self.play(FadeIn(note, report))

        txt = Tex("Age, sex, weight, height, etc.").scale(1.5).shift(2 * UP)

        self.play(Write(txt))

        self.wait(0.6)

        brain0 = (
            ImageMobject("images/brain0/slice68.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(4)
            .rotate(180 * DEGREES)
            .shift(3 * LEFT)
        )

        brain1 = (
            ImageMobject("images/brain1/slice68.png")
            .set_resampling_algorithm(RESAMPLING_ALGORITHMS["none"])
            .scale(4)
            .rotate(180 * DEGREES)
            .shift(3 * RIGHT)
        )

        self.play(FadeOut(note), FadeOut(report), FadeOut(txt), run_time=2)
        self.play(FadeIn(brain0), FadeIn(brain1), run_time=2)

        txt_male = Tex("Male ?").next_to(brain0, UP)
        txt_female = Tex("Female ? ").next_to(brain1, UP)

        self.play(Write(txt_male), Write(txt_female))

        self.wait(0.7)
        self.play(FadeOut(brain0, brain1, txt_male, txt_female))

        # Autoencoder
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

        brain0.next_to(autoencoder, direction=LEFT, buff=1.0)
        arrowin = Arrow(
            brain0.get_right(), autoencoder.get_left(), color=WHITE, buff=0.2
        )

        output_img = brain0.copy().next_to(autoencoder, RIGHT, buff=1.0)

        output = Tex("Male").next_to(autoencoder, DOWN, buff=1.0)
        arrowgender = Arrow(
            autoencoder.get_bottom(), output.get_top(), color=WHITE, buff=0.2
        )

        arrowout = Arrow(
            autoencoder.get_right(), output_img.get_left(), color=WHITE, buff=0.2
        )

        self.play(FadeIn(autoencoder))
        self.play(LaggedStart(FadeIn(brain0), GrowArrow(arrowin), lag_ratio=0.5))
        self.play(LaggedStart(GrowArrow(arrowout), FadeIn(output_img), lag_ratio=0.5))
        self.play(LaggedStart(GrowArrow(arrowgender), FadeIn(output), lag_ratio=0.5))
        self.wait(0.6)

        self.play(
            FadeOut(
                autoencoder, brain0, arrowin, arrowout, output, arrowgender, output_img
            )
        )
        # 2D Latent space
        self.next_section(skip_animations=False)

        axes = Axes(
            x_range=[0, 1, 0.5],
            y_range=[0, 1, 0.5],
            x_length=6,
            y_length=6,
            axis_config={
                "color": WHITE,
                "include_numbers": True,
                "include_tip": False,
                "font_size": 18,
            },
        )

        x_label = (
            Text("Dimension 1", font_size=16)
            .next_to(axes.x_axis, DOWN)
            .shift((axes.width / 2) * RIGHT)
        )
        y_label = (
            Text("Dimension 2", font_size=16)
            .next_to(axes.y_axis, LEFT)
            .shift((axes.height / 2) * UP)
        )

        mean1, cov1 = [2, 2], [[1, 0.5], [0.5, 1]]
        data1 = np.random.multivariate_normal(mean1, cov1, 100)

        mean2, cov2 = [7, 7], [[1, -0.5], [-0.5, 1]]
        data2 = np.random.multivariate_normal(mean2, cov2, 100)

        outliers1 = np.array([[8, 1], [9, 0], [7, 2]])
        outliers2 = np.array([[0, 8], [-1, 7], [2, 9]])

        data1 = np.concatenate([data1, outliers1])
        data2 = np.concatenate([data2, outliers2])

        data1_normalized = (data1 - np.min(data1, axis=0)) / (
            np.max(data1, axis=0) - np.min(data1, axis=0)
        )
        data2_normalized = (data2 - np.min(data2, axis=0)) / (
            np.max(data2, axis=0) - np.min(data2, axis=0)
        )

        dots = VGroup()

        for point in data1_normalized:
            dot = Dot(
                point=axes.c2p(point[0], point[1]),
                color=BLUE,
                fill_opacity=0.7,
                radius=0.03,
            )
            dots.add(dot)

        for point in data2_normalized:
            dot = Dot(
                point=axes.c2p(point[0], point[1]),
                color=RED,
                fill_opacity=0.7,
                radius=0.03,
            )
            dots.add(dot)

        legend_data = [("Male", BLUE), ("Female", RED)]

        legend_items = VGroup()
        for label, color in legend_data:
            dot = Dot(color=color)
            text = Text(label).scale(0.5)
            item = VGroup(dot, text).arrange(RIGHT, buff=0.2, aligned_edge=LEFT)
            legend_items.add(item)

        # Position the legend in the scene
        legend = (
            VGroup(*legend_items)
            .arrange(DOWN, buff=0.2, aligned_edge=LEFT)
            .to_corner(UR)
        )

        self.play(FadeIn(axes), Write(x_label), Write(y_label))
        self.wait(1)
        self.play(FadeIn(dots))
        self.play(FadeIn(legend))

        self.wait(0.8)

        self.play(FadeOut(axes, dots, legend, x_label, y_label))

        self.wait(1)

        txt = Tex("Limitations")
        self.play(Write(txt))

        self.wait(1.5)

        self.play(FadeOut(txt))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene4_1()
    scene.render()
