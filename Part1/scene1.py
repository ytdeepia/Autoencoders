from manim import *
from manim_voiceover import VoiceoverScene
import numpy as np
import matplotlib.pyplot as plt


class Scene1_1(VoiceoverScene, ThreeDScene):

    def create_prism(self, dimensions, fill_color, stroke_width):
        prism = Prism(
            dimensions=dimensions, fill_color=fill_color, stroke_width=stroke_width
        )
        prism.rotate(70 * DEGREES, axis=UP, about_point=ORIGIN).rotate(
            10 * DEGREES, axis=RIGHT, about_point=ORIGIN
        )
        return prism

    def construct(self):

        def create_layer(num_nodes, position_shift):
            layer = [
                Circle(radius=0.2, color=WHITE, stroke_width=2).shift(
                    0.4 * UP * i + position_shift
                )
                for i in range(num_nodes - 1, -num_nodes, -2)
            ]
            return layer

        def connect_layers(object, layer1, layer2):
            for dot1 in layer1:
                for dot2 in layer2:
                    start_point = dot1.get_center() + RIGHT * 0.2
                    end_point = dot2.get_center() + LEFT * 0.2
                    line = Line(start_point, end_point, color=WHITE, stroke_width=2)
                    object.add(line)

        self.wait(2)

        # Title

        title = Tex("Autoencoders").scale(1.5)
        underline = Underline(title)

        self.play(FadeIn(title))
        self.play(GrowFromEdge(underline, LEFT))

        self.wait(0.4)

        self.play(FadeOut(title), FadeOut(underline))

        latent_space = Axes(
            x_range=[0, 1, 0.5],
            y_range=[0, 1, 0.5],
            x_length=3,
            y_length=3,
            axis_config={
                "color": WHITE,
                "include_numbers": True,
                "include_tip": False,
                "font_size": 18,
            },
        )

        dot1 = Dot(latent_space.c2p(0.5, 0.5), color=RED)
        dot2 = Dot(latent_space.c2p(0.2, 0.4), color=BLUE)
        dot3 = Dot(latent_space.c2p(0.6, 0.3), color=GREEN)
        dot4 = Dot(latent_space.c2p(0.8, 0.2), color=ORANGE)

        self.play(FadeIn(latent_space, dot1, dot2, dot3, dot4))
        latent_title = (
            Tex("Latent Space").scale(0.8).next_to(latent_space, UP, buff=0.5)
        )

        self.play(Write(latent_title))

        self.wait(0.6)

        # Autoencoder
        autoencoder = VGroup()

        input_layer = create_layer(8, LEFT * 5 + 0.3 * DOWN)
        hidden_layer1 = create_layer(5, LEFT * 2 + 0.3 * DOWN)
        latent_layer = create_layer(3, ORIGIN + 0.3 * DOWN)
        hidden_layer2 = create_layer(5, RIGHT * 2 + 0.3 * DOWN)
        output_layer = create_layer(8, RIGHT * 5 + 0.3 * DOWN)

        for dot in (
            input_layer + hidden_layer1 + latent_layer + hidden_layer2 + output_layer
        ):
            autoencoder.add(dot)

        connect_layers(autoencoder, input_layer, hidden_layer1)
        connect_layers(autoencoder, hidden_layer1, latent_layer)
        connect_layers(autoencoder, latent_layer, hidden_layer2)
        connect_layers(autoencoder, hidden_layer2, output_layer)

        autoencoder.scale(0.8)

        self.play(FadeOut(latent_space, dot1, dot2, dot3, dot4, latent_title))
        self.play(FadeIn(autoencoder), run_time=2)

        self.wait(0.6)

        # UNet

        layer_1 = self.create_prism(
            dimensions=[3.5, 3.5, 0.3], fill_color=BLUE, stroke_width=1
        )
        layer_2 = self.create_prism(
            dimensions=[2, 2, 0.9], fill_color=BLUE, stroke_width=1
        )
        layer_3 = self.create_prism(
            dimensions=[1, 1, 0.6], fill_color=BLUE, stroke_width=1
        )
        layer_4 = self.create_prism(
            dimensions=[0.5, 0.5, 0.9], fill_color=BLUE, stroke_width=1
        )
        layer_5 = layer_3.copy()
        layer_6 = layer_2.copy()
        layer_7 = layer_1.copy()

        act_1 = self.create_prism(
            dimensions=[3.5, 3.5, 0.1], fill_color=GREEN, stroke_width=1
        )
        act_2 = self.create_prism(
            dimensions=[2, 2, 0.1], fill_color=GREEN, stroke_width=1
        )
        act_3 = self.create_prism(
            dimensions=[1, 1, 0.1], fill_color=GREEN, stroke_width=1
        )
        act_4 = self.create_prism(
            dimensions=[0.5, 0.5, 0.1], fill_color=GREEN, stroke_width=1
        )
        act_5 = act_3.copy()
        act_6 = act_2.copy()
        act_7 = act_1.copy()

        act_1.next_to(layer_1, RIGHT, buff=0.05)
        layer_2.next_to(act_1, RIGHT, buff=0.8)
        act_2.next_to(layer_2, RIGHT, buff=0.05)
        layer_3.next_to(act_2, RIGHT, buff=0.8)
        act_3.next_to(layer_3, RIGHT, buff=0.05)
        layer_4.next_to(act_3, RIGHT, buff=0.8)
        act_4.next_to(layer_4, RIGHT, buff=0.05)
        layer_5.next_to(act_4, RIGHT, buff=0.8)
        act_5.next_to(layer_5, RIGHT, buff=0.05)
        layer_6.next_to(act_5, RIGHT, buff=0.8)
        act_6.next_to(layer_6, RIGHT, buff=0.05)
        layer_7.next_to(act_6, RIGHT, buff=0.8)
        act_7.next_to(layer_7, RIGHT, buff=0.05)

        # Output image

        # Arrows

        arrow2 = Arrow(
            start=act_1.get_right(), end=layer_2.get_left(), color=WHITE, buff=0.05
        )
        arrow3 = Arrow(
            start=act_2.get_right(), end=layer_3.get_left(), color=WHITE, buff=0.05
        )
        arrow4 = Arrow(
            start=act_3.get_right(), end=layer_4.get_left(), color=WHITE, buff=0.05
        )
        arrow5 = Arrow(
            start=act_4.get_right(), end=layer_5.get_left(), color=WHITE, buff=0.05
        )
        arrow6 = Arrow(
            start=act_5.get_right(), end=layer_6.get_left(), color=WHITE, buff=0.05
        )
        arrow7 = Arrow(
            start=act_6.get_right(), end=layer_7.get_left(), color=WHITE, buff=0.05
        )

        unet = VGroup()
        unet.add(
            layer_1,
            layer_2,
            layer_3,
            layer_4,
            layer_5,
            layer_6,
            layer_7,
            act_1,
            act_2,
            act_3,
            act_4,
            act_5,
            act_6,
            act_7,
            arrow2,
            arrow3,
            arrow4,
            arrow5,
            arrow6,
            arrow7,
        )

        unet.scale(0.3).to_edge(RIGHT, buff=1.0)

        self.play(autoencoder.animate.scale(0.4).to_edge(LEFT, buff=1.0), run_time=1.5)

        self.play(FadeIn(unet), run_time=2)

        self.wait(0.5)

        # DALLE and Midjourney

        midjourney = SVGMobject(
            "svg/midjourney.svg",
            opacity=1,
            stroke_color=WHITE,
        ).scale(1.2)

        openai = SVGMobject(
            "svg/openai.svg",
            opacity=1,
            fill_opacity=1,
            stroke_color=WHITE,
            fill_color=WHITE,
        ).scale(0.8)

        openai.shift(3 * LEFT)
        midjourney.shift(3 * RIGHT)

        self.play(FadeOut(autoencoder), FadeOut(unet))
        self.play(FadeIn(openai))
        self.play(FadeIn(midjourney))

        self.wait(0.4)

        self.play(FadeOut(openai, midjourney))

        self.wait(2)


if __name__ == "__main__":
    scene = Scene1_1()
    scene.render()
