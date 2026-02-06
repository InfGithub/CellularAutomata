from manim import *

class MathExplanation(Scene):
    def construct(self):
        
        self.basic()
        self.example1()
        self.example2()
        self.example3()
        self.example4()
        self.example5()
        ...

    def basic(self):

        ECA = Text("Elementary Cellular Automata (ECA)", font_size=48)

        self.play(FadeIn(ECA))
        self.wait(1)
        self.play(
            ECA.animate.move_to(UP * 2.5),
            rate_func=smooth,
            run_time=1
        )

        ECAc = Text("初等元胞自动机", font_size=40)
        self.play(FadeIn(ECAc))
        self.wait(1)
        self.play(
            ECAc.animate.move_to(UP * 2),
            rate_func=smooth,
            run_time=1
        )

        self.wait(0.5)
        ECAs = Text("Elementary Cellular Automata", font_size=52)
        ECAs.move_to(ECA)
        self.play(FadeOut(ECAc), Transform(ECA, ECAs))

        intro = Text("ECA是一个由简单规则驱动的、理想化的计算系统，\n能够展现出从高度有序到完全混沌的丰富动态行为。", font_size=36)
        intro2 = Text("复杂的全局模式可以由大量遵循简单局部规则的\n元胞通过相互作用而涌现出来。", font_size=48)

        self.play(Write(intro))
        self.wait(3)
        self.play(Transform(intro, intro2))
        self.wait(3)

        self.play(FadeOut(intro), FadeOut(ECA))

    def example1(self):
        
        g1 = VGroup(
            Text("ECA是一个离散的动力系统，可以用一个四元组严格定义", font_size=32),
            MathTex("(L,S,N,f)", font_size=48),
            Text("：", font_size=32)
        )
        g1.arrange(RIGHT, buff=0.1)

        self.play(Write(g1))
        self.play(
            g1.animate.move_to(UP * 2.5),
            rate_func=smooth,
            run_time=1
        )
        g2 = VGroup(
            Text("元胞空间", font_size=30),
            MathTex("(L)", font_size=40),
            Text("：一个", font_size=30),
            MathTex("n", font_size=40),
            Text("维的无限网格。", font_size=30)
        )
        g2.arrange(RIGHT, buff=0.1)

        self.play(Write(g2))
        self.play(
            g2.animate.move_to(UP * 1.9),
            rate_func=smooth,
            run_time=1
        )
        g3 = VGroup(
            Text("状态集", font_size=30),
            MathTex("(S)", font_size=40),
            Text("：一个有限的、离散的状态集合。", font_size=30),
        )
        g3.arrange(RIGHT, buff=0.1)

        self.play(Write(g3))
        self.play(
            g3.animate.move_to(UP * 1.3),
            rate_func=smooth,
            run_time=1
        )
        g4 = VGroup(
            Text("邻居", font_size=30),
            MathTex("(N)", font_size=40),
            Text("：定义了中心元胞与周围哪些元胞互动。", font_size=30),
        )
        g4.arrange(RIGHT, buff=0.1)

        self.play(Write(g4))
        self.play(
            g4.animate.move_to(UP * 0.7),
            rate_func=smooth,
            run_time=1
        )
        g5 = VGroup(
            Text("局部规则", font_size=30),
            MathTex("(f)", font_size=40),
            Text("：局部转移函数接收一个来自", font_size=30),
            MathTex("S^{\\lvert N \\rvert}", font_size=40),
            Text("的向量", font_size=30)
        )
        g5.arrange(RIGHT, buff=0.1)

        self.play(Write(g5))
        self.play(
            g5.animate.move_to(UP * 0.1),
            rate_func=smooth,
            run_time=1
        )
        g6 = VGroup(
            Text("（其中", font_size=30),
            MathTex("\\lvert N \\rvert", font_size=40),
            Text("是邻居的数量），并输出中心元胞在下一时刻的状态。", font_size=30),
        )
        g6.arrange(RIGHT, buff=0.1)
        g6.move_to(DOWN * 0.5)
        self.play(Write(g6))
        g7 = VGroup(
            MathTex("f: S^{|N|} \\to S", font_size=40)
        )
        g7.arrange(RIGHT, buff=0.1)
        g7.move_to(DOWN * 1.1)
        self.play(Write(g7))

        self.wait(3)
        self.play(
            FadeOut(g1),
            FadeOut(g2),
            FadeOut(g3),
            FadeOut(g4),
            FadeOut(g5),
            FadeOut(g6),
            FadeOut(g7))

    def example2(self):
        t1 = Text("工作原理与全局演化", font_size=48)
        t1.move_to(UP * 3)
        self.play(Write(t1))
        self.wait(1)
        
        g1 = VGroup(
            Text("构型：在时间刻", font_size=30),
            MathTex("t", font_size=40),
            Text("，整个n维空间的状态是一个构型", font_size=30),
            MathTex("c^t", font_size=40),
            Text("。", font_size=30),
        )
        g1.arrange(RIGHT, buff=0.1)
        g1.move_to(UP * 2)
        self.play(Write(g1))
        
        g2 = VGroup(
            Text("全局函数", font_size=30),
            MathTex("\\Phi", font_size=40),
            Text("：将当前构型", font_size=30),
            MathTex("c^t", font_size=40),
            Text("映射到下一构型", font_size=30),
            MathTex("c^{t+1}", font_size=40),
            Text("。", font_size=30),
        )
        g2.arrange(RIGHT, buff=0.1)
        g2.move_to(UP * 1.5)
        self.play(Write(g2))
        
        g3 = VGroup(
            MathTex("\\Phi(c^t) = c^{t+1}", font_size=40)
        )
        g3.arrange(RIGHT, buff=0.1)
        g3.move_to(UP * 1)
        self.play(Write(g3))
        
        g4 = VGroup(
            Text("更新公式：对于空间中的每一个位置", font_size=30),
            MathTex("\\vec{x}", font_size=40),
            Text("（一个n维坐标向量），新状态由下式给出：", font_size=30),
        )
        g4.arrange(RIGHT, buff=0.1)
        g4.move_to(UP * 0.5)
        self.play(Write(g4))
        
        g5 = VGroup(
            MathTex("[\\Phi(c)]_{\\vec{x}} = f( \\{ c_{\\vec{x} + \\vec{n}} \\mid \\vec{n} \\in N \\}", font_size=40)
        )
        g5.arrange(RIGHT, buff=0.1)
        self.play(Write(g5))
        
        g6 = VGroup(
            Text("其中", font_size=30),
            MathTex("N", font_size=40),
            Text("是邻居向量的集合。这个公式意味着，", font_size=30),
        )
        g6.arrange(RIGHT, buff=0.1)
        g6.move_to(DOWN * 0.5)
        self.play(Write(g6))
        
        g7 = VGroup(
            Text("新构型在位置", font_size=30),
            MathTex("\\vec{x}", font_size=40),
            Text("的值，由旧构型中", font_size=30),
            MathTex("\\vec{x}", font_size=40),
            Text("的所有邻居的状态共同决定。", font_size=30),
        )
        g7.arrange(RIGHT, buff=0.1)
        g7.move_to(DOWN * 1)
        self.play(Write(g7))
        self.wait(3)
        self.play(
            FadeOut(g1),
            FadeOut(g2),
            FadeOut(g3),
            FadeOut(g4),
            FadeOut(g5),
            FadeOut(g6),
            FadeOut(g7),
            FadeOut(t1))

    def example3(self):

        t0 = Text("以下是相关示例，可帮助你理解。", font_size=56)
        self.play(Write(t0))
        self.wait(3)
        self.play(FadeOut(t0))

        t1 = Text("元胞空间", font_size=56)
        self.play(Write(t1))
        self.wait(1)

        matrix1 = MathTex("\\begin{bmatrix} 0 & 1 & 1 & 0 & 1 \\end{bmatrix}_{L}", font_size=60)
        matrix2 = MathTex("\\begin{bmatrix} ? & ? & ? & ? & ? \\end{bmatrix}_{N}", font_size=60)

        m1 = MathTex("n = 1", font_size=60)

        matrix1.move_to(UP*2.5)
        matrix2.move_to(DOWN*2.5)
        m1.move_to(LEFT * 5 + UP * 2.5)

        g1 = VGroup(matrix1, matrix2, m1)
        self.play(Transform(t1, g1))
        self.wait(1)

        t2 = Text("状态集", font_size=56)
        self.wait(1)
        self.play(Write(t2))

        m2 = MathTex("S=\\{0,1\\}", font_size=60)
        m2.move_to(LEFT * 5)
        self.play(Transform(t2, m2))
        self.wait(1)

        t3 = Text("邻居", font_size=56)
        self.play(Write(t3))
        self.wait(1)

        t4 = Text("Moore", font_size=56)
        self.play(Transform(t3, t4))
        self.wait(1)

        matrix3 = MathTex("\\begin{pmatrix} i-1 & i & i+1 \\end{pmatrix}", font_size=48)
        matrix3.move_to(LEFT * 5 + DOWN * 2.5)
        self.play(Transform(t3, matrix3))

        t5 = Text("局部规则", font_size=56)
        self.play(Write(t5))
        self.wait(1)

        t8 = Text("Wolfram", font_size=56)
        self.play(Transform(t5, t8))
        self.wait(1)

        t6 = Text("查表法", font_size=56)
        self.play(Transform(t5, t6))
        self.wait(1)

        m3 = MathTex("Rule: 30", font_size=56)
        m3.move_to(RIGHT * 5 + UP * 2.5)

        m4 = MathTex("0b00011110", font_size=48)
        m4.move_to(RIGHT * 5 + UP * 2.5)

        self.play(Transform(t5, m3))
        self.wait(1)
        self.play(Transform(t5, m4))

        m5 = MathTex("000: 0", font_size=42)
        m6 = MathTex("001: 1", font_size=42)
        m7 = MathTex("010: 1", font_size=42)
        m8 = MathTex("011: 1", font_size=42)
        m9 = MathTex("100: 1", font_size=42)
        m10 = MathTex("101: 0", font_size=42)
        m11 = MathTex("110: 0", font_size=42)
        m12 = MathTex("111: 0", font_size=42)

        m5.move_to(RIGHT * 5 + UP * 1.5)
        m6.move_to(RIGHT * 5 + UP * 0.8)
        m7.move_to(RIGHT * 5 + UP * 0.1)
        m8.move_to(RIGHT * 5 + DOWN * 0.6)
        m9.move_to(RIGHT * 5 + DOWN * 1.3)
        m10.move_to(RIGHT * 5 + DOWN * 2)
        m11.move_to(RIGHT * 5 + DOWN * 2.7)
        m12.move_to(RIGHT * 5 + DOWN * 3.4)

        g2 = VGroup(
            m5,
            m6,
            m7,
            m8,
            m9,
            m10,
            m11,
            m12
        )
        self.play(Transform(m4, g2))

        self.play(Transform(t5, m3))

        self.play(
            FadeOut(t1),
            FadeOut(t2),
            FadeOut(t3),
            FadeIn(matrix1),
            FadeIn(matrix2))

        self.play(
            t5.animate.move_to(LEFT * 5 + UP * 2.5),
            rate_func=smooth,
            run_time=1
        )
        self.play(
            m4.animate.move_to(RIGHT * 5),
            rate_func=smooth,
            run_time=1
        )

        self.play(
            m4.animate.scale(1.2),
            rate_func=smooth,
            run_time=1
        )

        t7 = Text("开始迭代", font_size=56)
        self.play(Write(t7))
        self.wait(1)
        self.play(FadeOut(t7))

        #---------------------------------------------

        p = [
            "\\begin{bmatrix} 0 & ? & ? & ? & ? \\end{bmatrix}_{N}",
            "\\begin{bmatrix} 0 & 1 & ? & ? & ? \\end{bmatrix}_{N}",
            "\\begin{bmatrix} 0 & 1 & 0 & ? & ? \\end{bmatrix}_{N}",
            "\\begin{bmatrix} 0 & 1 & 0 & 0 & ? \\end{bmatrix}_{N}",
            "\\begin{bmatrix} 0 & 1 & 0 & 0 & 1 \\end{bmatrix}_{N}"
        ]

        q = [5, 3, 6, 5, 2]

        for i in [0, 1, 2, 3, 4]:

            matrix1_elements = matrix1[0][1:6]
            matrix2_elements = matrix2[0][1:6]

            highlight_boxes = VGroup()
            if i == 0:
                highlight_boxes.add(SurroundingRectangle(
                    VGroup(matrix1_elements[0], matrix1_elements[1]),
                    color=RED,
                    buff=0.1,
                    stroke_width=3
                ))
                highlight_boxes.add(SurroundingRectangle(
                    matrix1_elements[4],
                    color=RED,
                    buff=0.1,
                    stroke_width=3
                ))
            elif i == 4:
                highlight_boxes.add(SurroundingRectangle(
                    matrix1_elements[0],
                    color=RED,
                    buff=0.1,
                    stroke_width=3
                ))
                
                highlight_boxes.add(SurroundingRectangle(
                    VGroup(matrix1_elements[3], matrix1_elements[4]),
                    color=RED,
                    buff=0.1,
                    stroke_width=3
                ))
            else:
                highlight_boxes.add(SurroundingRectangle(
                    VGroup(*[matrix1_elements[i % 5] for i in [i-1, i, i+1]]),
                    color=RED,
                    buff=0.1,
                    stroke_width=3
                ))

            highlight_boxes.add(SurroundingRectangle(
                matrix2_elements[i],
                color=BLUE,
                buff=0.1,
                stroke_width=3
            ))

            h1 = m4[q[i]]

            highlight_boxes.add(SurroundingRectangle(
                h1[0:3],
                color=ORANGE,
                buff=0.1,
                stroke_width=3
            ))

            self.play(Create(highlight_boxes))

            merge_arrow = Arrow(
                start=matrix1_elements[i].get_bottom(),
                end=matrix2_elements[i].get_top(),
                color=GREEN,
                buff=0.2,
                stroke_width=4
            )

            self.play(GrowArrow(merge_arrow))

            updated_matrix2 = MathTex(p[i], font_size=60)
            updated_matrix2.move_to(DOWN * 2.5)

            self.play(Transform(matrix2, updated_matrix2))
            self.play(
                FadeOut(highlight_boxes),
                FadeOut(merge_arrow),
            )

            self.wait(1)






        self.wait(3)


        self.play(
            FadeOut(
                VGroup(
                    matrix1, matrix2, t5, m4
                )
            )
        )

    def example4(self):

        title = Text("一维Rule30的时空演化", font_size=48, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        initial_state = [0, 0, 0, 0, 1, 0, 0, 0, 0]
        states = [initial_state]
        
        for _ in range(6):
            new_state = []
            for i in range(len(initial_state)):
                left = states[-1][(i-1) % len(initial_state)]
                center = states[-1][i]
                right = states[-1][(i+1) % len(initial_state)]
                
                if (left, center, right) in [(1,1,1), (1,1,0), (1,0,1), (0,0,0)]:
                    new_state.append(0)
                else:
                    new_state.append(1)
            states.append(new_state)

        cell_size = 0.4
        grid = VGroup()
        
        for t, state in enumerate(states):
            row = VGroup()
            for i, cell in enumerate(state):
                square = Square(side_length=cell_size, stroke_width=1)
                if cell == 1:
                    square.set_fill(BLACK, opacity=1)
                else:
                    square.set_fill(WHITE, opacity=1)
                square.move_to([(i - len(state)//2) * cell_size, 
                            (3 - t) * cell_size, 0])
                row.add(square)
            grid.add(row)

        grid.move_to(ORIGIN)
        self.play(Create(grid), run_time=2)
        self.wait(2)

        explanation = Text("Rule30产生看似随机的混沌模式", font_size=32)
        explanation.next_to(grid, DOWN, buff=0.5)
        self.play(Write(explanation))
        self.wait(2)


        self.play(
            FadeOut(title),
            FadeOut(grid),
            FadeOut(explanation),
        )

    def example5(self):
        t0 = Text("现在，让我们来点更有意思的......", font_size=56)
        self.play(Write(t0))
        self.wait(3)
        self.play(FadeOut(t0))

if __name__ == "__main__":
        
    config.quality = "fourk_quality"
    config.pixel_height = 2160
    config.pixel_width = 3840
    config.frame_rate = 120
    scene = MathExplanation()
    scene.render()

