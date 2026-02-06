from manim import *

class MathExplanation(ThreeDScene):
    def construct(self):
        self.basic()
        self.example1()
        self.example2()
        self.example3()
        self.example4()
        self.example5()
        self.example6()

    def basic(self):
        x = Text("ECA在Python中的实现", font_size=72).move_to(UP*1.5)
        y = Text("作者の程序 | 依赖Numpy", font_size=56).move_to(DOWN*2)
        self.play(AnimationGroup(
            Write(x),
            Write(y),
        ))
        self.wait(2)
        self.play(AnimationGroup(
            FadeOut(x),
            FadeOut(y),
        ))
        
        k = VGroup(
            Text("定义我们所讨论的ECA", font_size=48),
            MathTex("(L,S,N,f)", font_size=72),
            Text("：", font_size=48)
        ).arrange(RIGHT, buff=0.1).move_to(UP*2.5)
        self.play(AnimationGroup(
            Write(k),
        ))
        self.wait(2)
        x = VGroup(
            Text("元胞空间", font_size=32),
            MathTex("(L)", font_size=48),
            Text("：", font_size=32),
            MathTex("D", font_size=48),
            Text("阶张量，尺寸由参数所定", font_size=32),
        ).arrange(RIGHT, buff=0.1).move_to(UP*1.5)
        y = VGroup(
            Text("状态集", font_size=32),
            MathTex("(S)", font_size=48),
            Text("：", font_size=32),
            MathTex("N^D", font_size=48),
            Text("阶", font_size=32),
            MathTex("M", font_size=48),
            Text("维张量（即规则空间）", font_size=32),
        ).arrange(RIGHT, buff=0.1).move_to(UP*0.9)
        z = VGroup(
            Text("邻居", font_size=32),
            MathTex("(N)", font_size=48),
            Text("：", font_size=32),
            Text("摩尔邻居，", font_size=32),
            MathTex("3^N", font_size=48),
            Text("窗口", font_size=32),
        ).arrange(RIGHT, buff=0.1).move_to(UP*0.3)
        w = VGroup(
            Text("局部规则", font_size=32),
            MathTex("(f)", font_size=48),
            Text("：", font_size=32),
            Text("规则空间LUT", font_size=32),
        ).arrange(RIGHT, buff=0.1).move_to(DOWN*0.3)
        v = VGroup(
            Text("其中，", font_size=32),
            MathTex("D, N, M", font_size=48),
            Text("分别是维度参数、邻居范围、状态数量。", font_size=32),
        ).arrange(RIGHT, buff=0.1).move_to(DOWN*1.3)
        u = VGroup(
            Text("作简：", font_size=32),
            MathTex("N, M", font_size=48),
            Text("取", font_size=32),
            MathTex("3, 2", font_size=48),
        ).arrange(RIGHT, buff=0.1).move_to(DOWN*1.9)
        self.play(AnimationGroup(
            FadeIn(x, y, z, w, v, u)
        ))
        self.wait(4)
        
        self.play(AnimationGroup(
            FadeOut(x, y, z, w, v, u, k)
        ))
    def example1(self):
        x = Text("让我们先从元胞空间入手。", font_size=48)
        self.play(FadeIn(x))
        self.wait(2)
        
        y = VGroup(
            Text("元胞空间是一个", font_size=32),
            MathTex("D", font_size=48),
            Text("阶张量", font_size=32),
            MathTex("C", font_size=48),
        ).arrange(RIGHT, buff=0.1).move_to(UP*3)
        z = VGroup(
            Text("其中，每个元素代表一个元胞，元素值即该元胞的状态。", font_size=32),
        ).arrange(RIGHT, buff=0.1).move_to(UP*2.5)

        self.play(AnimationGroup(
            Transform(x, VGroup(
                y, z
            )),
        ))
        self.wait(2)

        data = [
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1]
        ]
        w = Matrix(data).move_to(LEFT*5 + DOWN*5)
        code = "size = (3, 3, 3)\ncell = np.zeros(size, dtype=np.uint8)"
        v = Code(
            code_string=code,
            language="numpy",
            formatter_style="github-dark",
            background="window",
            paragraph_config={"font": "Consolas"}
        ).move_to(RIGHT*5 + DOWN*5)

        self.play(AnimationGroup(
            w.animate.move_to(LEFT*5),
            v.animate.move_to(RIGHT*2.5),
            run_time=1
        ))
        a = MathTex("Dim=2", font_size=48).move_to(LEFT*5 + DOWN*2)
        b = MathTex("Dim=3", font_size=48).move_to(RIGHT*2.5 + DOWN*2)
        c = VGroup(
            Text("状态数量", font_size=32),
            MathTex("M=2", font_size=48),
        ).arrange(RIGHT, buff=0.1).move_to(DOWN*3)
        d = MathTex("\\{0, 1\\}", font_size=48).move_to(DOWN*3+RIGHT*3)
        self.play(
            FadeIn(a),
            FadeIn(b),
            FadeIn(c),
            Write(d)
        )
        self.wait(4)
        self.play(
            FadeOut(x, w, v, a, b, c, d)
        )

    def example2(self):

        k = VGroup(
            Text("邻居范围", font_size=32),
            MathTex("N=3", font_size=48),
        ).arrange(RIGHT, buff=0.1).move_to(UP*3)
        self.play(FadeIn(k))
        self.wait(1)

        data = [
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ]
        x = Matrix(data, h_buff=1.0)
        self.play(Write(x))
        self.wait(3)
        
        elements = x.get_entries()
        indices = [
            6,  7,  8,
            11, 12, 13,
            16, 17, 18
        ]
        cells = VGroup(*[elements[i] for i in indices])
        y = SurroundingRectangle(cells, color=BLUE, buff=0.1)
        self.play(Create(y))
        self.wait(3)

        z = Text("这是一个滑动窗口。", font_size=32).move_to(DOWN*2.5)
        self.play(Write(z))
        self.wait(1)
        w = Text("将其展平，用于索引。", font_size=32).move_to(DOWN*3.0)
        self.play(Write(w))
        self.wait(1)

        data = [[
            0, 1, 0,
            1, 0, 1,
            0, 1, 0
        ]]
        v = Matrix(data, h_buff=1.0)
        a = VGroup(k, x, y)
        self.play(
            Transform(a, v)
        )
        self.wait(1)
        b = VGroup(z, w)
        c = Text("将这个向量作为某个函数的输入，", font_size=32).move_to(DOWN*2.5)
        d = Text("以得到下一刻的元胞状态。", font_size=32).move_to(DOWN*3)
        e = VGroup(c, d)
        self.play(
            Transform(b, e)
        )
        self.wait(1)
        f = MathTex("f(x)=?", font_size=48).move_to(DOWN*1.5)
        self.play(
            Write(f)
        )
        self.wait(2)
        self.play(
            FadeOut(
                f, b, e, a
            )
        )

    def example3(self):
        data = [
            [0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0, 0, 1, 0],
        ]
        x = VGroup(
            *[Matrix([item], h_buff=0.5).move_to(UP*(3 - index)+LEFT*3.5) for index, item in enumerate(data)]
        )
        y = MathTex("Dim=2, Length: 3^2=9", font_size=48).move_to(DOWN*3+LEFT*3.5)
        self.play(
            FadeIn(x, y)
        )
        self.wait(2)
        z = Text("观察可得，这些向量可以作为一个", font_size=32).move_to(RIGHT*3.5+UP*0.5)
        w = VGroup(
            MathTex("9", font_size=48),
            Text("阶", font_size=32),
            MathTex("2", font_size=48),
            Text("维张量的索引。", font_size=32),
        ).arrange(RIGHT, buff=0.1).move_to(RIGHT*3.5)
        k = Text("对张量进行索引即可得到元胞状态。", font_size=32).move_to(RIGHT*3.5+DOWN*0.5)
        v = VGroup(
            z, w, k
        )
        self.play(
            Write(v)
        )
        self.wait(2)
        a = Text("如何构造这样一个张量呢？", font_size=36).move_to(RIGHT*3.5)

        self.play(
            Transform(v, a)
        )
        self.wait(3)

        self.play(
            FadeOut(x, y, v)
        )
        
    def example4(self):
        x = MathTex("N=3, M=2, Dim=1, Rule=30", font_size=48).move_to(UP*3.5)
        y = Text("给定条件", font_size=36).move_to(RIGHT*4 + DOWN*3)
        self.play(
            Write(x),
            Write(y)
        )
        self.wait(2)

        z = MathTex("30", font_size=64)
        self.play(
            FadeIn(z)
        )
        self.wait(2)

        self.play(
            Transform(
                z, MathTex("(00011110)_2", font_size=64),
            ),
            Transform(
                y, Text("转二进制", font_size=36).move_to(RIGHT*4 + DOWN*3)
            ),
        )
        self.wait(2)

        self.play(
            Transform(
                z, MathTex("(01111000)_2", font_size=64),
            ),
            Transform(
                y, Text("反转为小端序", font_size=36).move_to(RIGHT*4 + DOWN*3)
            ),
        )
        self.wait(2)

        self.play(
            Transform(
                y, Text("此时，即可对字符串进行索引。", font_size=36).move_to(RIGHT*3 + DOWN*3)
            ),
        )
        self.wait(2)

        w = VGroup(
            *[MathTex(f"C_{"{"}{index:03b}{"}"}={item}", font_size=48).move_to(UP*(2 - index*0.7)+LEFT*3.5) for index, item in enumerate("01111000")]
        )

        self.play(
            Transform(
                z, w
            ),
            Transform(
                y, Text("将其对应到张量的索引。", font_size=36).move_to(RIGHT*4 + DOWN*3)
            ),
        )
        self.wait(2)

        self.move_camera(
            phi=50*DEGREES,
            theta=-90*DEGREES,
            zoom=0.8,
            run_time=4,
            rate_func=smooth
        )
        values = [0, 1, 1, 1, 1, 0, 0, 0]

        self.begin_ambient_camera_rotation(rate=0.25)
        pos = [
            [ 0,  0,  0],
            [ 1,  0, -3],
            [ 4,  1,  5],
            [-3,  1,  3],
            [-4, -3, -2],
            [-4, -3,  2],
        ]
        element_list = list()
        for tick in range(6):
            positions = [
                [pos[tick][0] + dx, pos[tick][1] + dy, pos[tick][2] + dz]
                for dx in (-0.5, 0.5)
                for dy in (-0.5, 0.5)
                for dz in (-0.5, 0.5)
            ]
            elements = [VGroup(
                Cube(side_length=0.6, fill_color=BLUE).move_to(positions[index]),
                Text(str(values[index]), font_size=20).move_to(positions[index])
            ) for index in range(8)]
            element_list.append(elements)

            self.play(*[Create(elem) for elem in elements])
            for index in range(8):
                if values[index]:
                    self.play(
                        elements[index][0].animate.set_fill(GREEN),
                        run_time=0.5
                    )
            self.wait(5)
        self.stop_ambient_camera_rotation()
        for elements in element_list:
            self.play(
                FadeOut(*elements)
            )
        self.move_camera(
            phi=0*DEGREES,
            theta=-90*DEGREES,
            zoom=1,
            run_time=4,
            rate_func=smooth
        )
        self.wait(2)
        self.play(
            FadeOut(x, y, z, w)
        )
        n = 2**9
        b = VGroup(
            *[MathTex(f"C_{"{"}{pow(index, 9)%n:09b}{"}"}={item}", font_size=48).move_to(UP*(5 - index*0.5)+LEFT*3.5) for index, item in enumerate("1010101000010010101010")]
        )
        n = 2**27
        c = VGroup(
            *[MathTex(f"C_{"{"}{pow(index, 27)%n:027b}{"}"}={item}", font_size=48).move_to(UP*(5 - index*0.5)+RIGHT*2) for index, item in enumerate("0111100010010101010011")]
        )
        self.play(
            Write(b),
            Write(c)
        )
        self.wait(5)
        self.play(
            FadeOut(b, c)
        )

        v = VGroup(
            Text("延拓，即可构建", font_size=32),
            MathTex("9", font_size=48),
            Text("阶、", font_size=32),
            MathTex("27", font_size=48),
            Text("阶等", font_size=32),
            MathTex("2", font_size=48),
            Text("维的张量。", font_size=32),
        ).arrange(RIGHT, buff=0.1).move_to(UP*2.5)
        code = "value: int = 3 ** dim\nrule_repr = format(rule, f\"0{2 ** value}b\")[::-1].encode(\"ascii\")\nrepr_array = np.frombuffer(rule_repr, dtype=np.uint8) - 48\nrule_space = repr_array.reshape([2] * value)"
        u = Code(
            code_string=code,
            language="numpy",
            formatter_style="github-dark",
            background="window",
            paragraph_config={"font": "Consolas"}
        ).move_to(DOWN*0.5)
        a = VGroup(
            v, u
        )
        self.play(
            Write(a)
        )
        self.wait(5)
        self.play(
            FadeOut(a)
        )

    def example5(self):
        data = [
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ]
        x = Matrix(data, h_buff=1.0).move_to(LEFT*3.5)
        self.play(FadeIn(x))
        
        elements = x.get_entries()
        indices = [
            6,  7,  8,
            11, 12, 13,
            16, 17, 18
        ]
        cells = VGroup(*[elements[i] for i in indices])
        y = SurroundingRectangle(cells, color=BLUE, buff=0.1)
        self.play(Create(y))
        self.wait(1)

        data = [[
            0, 1, 0,
            1, 0, 1,
            0, 1, 0
        ]]
        z = Matrix(data, h_buff=0.4).move_to(RIGHT*3 + UP*3)

        self.play(
            Create(z)
        )
        self.wait(1)

        w = MathTex("C_{010101010}=k", font_size=48).move_to(RIGHT*3)

        self.play(
            Write(w)
        )
        self.wait(1)
        
        data = [
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, "k", 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ]
        a = Matrix(data, h_buff=1.0).move_to(LEFT*3.5)
        self.play(
            Transform(x, a)
        )

        v = Text("如此，就可以进行迭代。", font_size=36).move_to(RIGHT*3 + DOWN*3)
        self.play(
            FadeIn(v)
        )
        self.wait(4)

        self.play(
            FadeOut(x, y, z, w, v)
        )
        
        code = "padded = np.pad(cell, 1, mode=\"wrap\")\nwindows = as_strided(\n    padded,\n    shape=cell.shape + (3,) * ndim,\n    strides=padded.strides * 2\n)\nflat_windows = windows.reshape(-1, 3 ** ndim)\ncell.ravel()[:] = rule_space[tuple(flat_windows.T)]"
        u = Code(
            code_string=code,
            language="numpy",
            formatter_style="github-dark",
            background="window",
            paragraph_config={"font": "Consolas"}
        )
        a = Text("pad处理边缘情况，as_strided内存操作滑动窗口，\nreshape展平，最后索引更新状态。", font_size=36).move_to(DOWN*3)
        self.play(
            Write(u),
            Write(a)
        )
        self.wait(5)
        self.play(
            FadeOut(u, a)
        )

    def example6(self):
        x = Text("此外，还有镜像变换、补码变换函数。", font_size=36).move_to(UP*3)
        self.play(
            Write(x)
        )
        self.wait(2)
        code = "value = 3 ** dim\nrule_space = rule_space.transpose(range(value)[::-1])\nrule_space = np.flip(1 - rule_space)"
        y = Code(
            code_string=code,
            language="numpy",
            formatter_style="github-dark",
            background="window",
            paragraph_config={"font": "Consolas"}
        )
        self.play(
            Write(y)
        )
        self.wait(5)
        z = Text("为提升性能，还可将Numpy平替成Cupy。", font_size=36).move_to(DOWN*1.5)
        self.play(
            Write(z)
        )
        self.wait(3)
        w = Text("Github - https://github.com/InfGithub/CellularAutomata", font_size=36).move_to(DOWN*3)
        self.play(
            Write(w)
        )
        self.wait(5)
        self.play(
            FadeOut(x, y, z, w)
        )

if __name__ == "__main__":
    # config.disable_caching = True

    config.quality = "fourk_quality"
    config.pixel_height = 2160
    config.pixel_width = 3840
    config.frame_rate = 120

    scene = MathExplanation()
    scene.render()