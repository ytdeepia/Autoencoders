from manim import *
from manim_voiceover import VoiceoverScene

import numpy as np
import matplotlib.pyplot as plt


class Scene3_2(VoiceoverScene):
    def construct(self):

        self.wait(2)

        # Display the encoder with 2 neurons
        self.next_section(skip_animations=False)

        encoder = Polygon(
            [-1, 1, 0], [1, 0.4, 0], [1, -0.4, 0], [-1, -1, 0], color=PURPLE
        )
        encoder_txt = (
            Tex("Encoder", color=WHITE).scale(0.6).move_to(encoder.get_center())
        )

        bottleneck_txt = Tex("Bottleneck", color=WHITE).scale(0.6)
        bottleneck = SurroundingRectangle(bottleneck_txt, buff=0.3, color=BLUE)
        bottleneck_g = VGroup(bottleneck, bottleneck_txt)
        bottleneck_g.next_to(encoder, direction=RIGHT, buff=0.4)

        decoder = Polygon(
            [-1, 0.4, 0], [1, 1, 0], [1, -1, 0], [-1, -0.4, 0], color=PURPLE
        ).next_to(bottleneck, direction=RIGHT, buff=0.4)
        decoder_txt = (
            Tex("Decoder", color=WHITE).scale(0.6).move_to(decoder.get_center())
        )

        autoencoder = VGroup(
            encoder, bottleneck, decoder, encoder_txt, bottleneck_txt, decoder_txt
        )
        autoencoder.move_to(ORIGIN).scale(0.9)

        self.play(FadeIn(autoencoder))

        self.wait(0.4)

        neurons = VGroup(
            Circle(radius=0.3, color=WHITE),
            Circle(radius=0.3, color=WHITE),
        )

        neurons.arrange(DOWN, buff=0.4).next_to(encoder, RIGHT, buff=1.5)
        line1 = Line(
            encoder.get_corner(UR) + 0.4 * RIGHT + 0.5 * DOWN,
            neurons[0].get_left(),
            color=WHITE,
        )
        line2 = Line(
            encoder.get_corner(DR) + 0.4 * RIGHT + 0.5 * UP,
            neurons[0].get_left(),
            color=WHITE,
        )
        line3 = Line(
            encoder.get_corner(UR) + 0.4 * RIGHT + 0.5 * DOWN,
            neurons[1].get_left(),
            color=WHITE,
        )
        line4 = Line(
            encoder.get_corner(DR) + 0.4 * RIGHT + 0.5 * UP,
            neurons[1].get_left(),
            color=WHITE,
        )

        line1.add_updater(
            lambda l: l.put_start_and_end_on(
                encoder.get_corner(UR) + 0.4 * RIGHT + 0.5 * DOWN, neurons[0].get_left()
            )
        )
        line2.add_updater(
            lambda l: l.put_start_and_end_on(
                encoder.get_corner(DR) + 0.4 * RIGHT + 0.5 * UP, neurons[0].get_left()
            )
        )
        line3.add_updater(
            lambda l: l.put_start_and_end_on(
                encoder.get_corner(UR) + 0.4 * RIGHT + 0.5 * DOWN, neurons[1].get_left()
            )
        )
        line4.add_updater(
            lambda l: l.put_start_and_end_on(
                encoder.get_corner(DR) + 0.4 * RIGHT + 0.5 * UP, neurons[1].get_left()
            )
        )

        lines = VGroup(line1, line2, line3, line4)

        fc_layer = VGroup(lines, neurons)

        self.play(FadeOut(decoder, decoder_txt, bottleneck, bottleneck_txt), run_time=2)
        self.play(FadeIn(fc_layer))

        self.wait(0.2)

        random_numbers = np.random.randint(1, 100, size=(2, 1))
        features = [[str(num) for num in row] for row in random_numbers]
        features = Matrix(features).next_to(neurons, RIGHT, buff=1.5)
        latent_title = (
            Tex("Latent vector", color=WHITE).scale(0.8).next_to(features, UP, buff=1)
        )

        self.wait()

        self.wait(0.4)

        self.play(FadeIn(features, latent_title), run_time=1)

        self.wait(0.3)

        # Display the encoder with 3 neurons and more
        self.next_section(skip_animations=False)

        self.play(neurons.animate.shift(0.5 * UP))

        circle = Circle(radius=0.3, color=WHITE).next_to(neurons[1], DOWN, buff=0.4)
        neurons.add(circle)

        line5 = Line(
            encoder.get_corner(UR) + 0.4 * RIGHT + 0.5 * DOWN,
            neurons[2].get_left(),
            color=WHITE,
        )
        line6 = Line(
            encoder.get_corner(DR) + 0.4 * RIGHT + 0.5 * UP,
            neurons[2].get_left(),
            color=WHITE,
        )
        line5.add_updater(
            lambda l: l.put_start_and_end_on(
                encoder.get_corner(UR) + 0.4 * RIGHT + 0.5 * DOWN,
                circle.get_left(),
            )
        )
        line6.add_updater(
            lambda l: l.put_start_and_end_on(
                encoder.get_corner(DR) + 0.4 * RIGHT + 0.5 * UP,
                circle.get_left(),
            )
        )

        lines.add(line5, line6)

        random_numbers = np.random.randint(1, 100, size=(3, 1))
        features_2 = [[str(num) for num in row] for row in random_numbers]
        features_2 = Matrix(features_2).next_to(neurons, RIGHT, buff=1.5)

        self.play(
            LaggedStart(
                FadeIn(neurons[2], line5, line6),
                Transform(features, features_2),
                lag_ratio=0.2,
            ),
            run_time=1.5,
        )

        self.wait(0.4)

        self.play(
            FadeOut(features, latent_title),
            run_time=1,
        )

        dots = (
            VGroup(Dot(color=WHITE), Dot(color=WHITE), Dot(color=WHITE))
            .arrange(DOWN, buff=0.1)
            .next_to(encoder, RIGHT, buff=1.7)
        )

        brace = Brace(neurons, direction=RIGHT)
        brace_txt = brace.get_text("Any number of neurons").scale(0.6)

        self.play(
            FadeOut(neurons[1], line3, line4),
            FadeIn(dots),
            run_time=2,
        )

        self.play(FadeIn(brace, brace_txt), run_time=2)

        self.wait(0.4)

        lines.remove(line3, line4)
        neurons.remove(neurons[1])

        self.play(
            FadeOut(encoder, encoder_txt, fc_layer, dots, brace, brace_txt),
            run_time=1,
        )

        self.wait(2)


if __name__ == "__main__":
    scene = Scene3_2()
    scene.render()
