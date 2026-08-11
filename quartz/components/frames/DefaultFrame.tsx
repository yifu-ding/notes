import { PageFrame, PageFrameProps } from "./types"
import HeaderConstructor from "../Header"

const Header = HeaderConstructor()

/**
 * The default page frame — three-column layout with left sidebar, center
 * content (header + body + afterBody), and right sidebar, followed by a footer.
 *
 * This is the original Quartz layout, extracted from renderPage.tsx.
 */
export const DefaultFrame: PageFrame = {
  name: "default",
  render({
    componentData,
    header,
    beforeBody,
    pageBody: Content,
    afterBody,
    left,
    right,
    footer,
  }: PageFrameProps) {
    return (
      <>
        <div class="left sidebar">
          {left.map((BodyComponent) => (
            <BodyComponent {...componentData} />
          ))}
        </div>
        <div class="center">
          <div class="page-header">
            {componentData.fileData.filePath &&
              (() => {
                const rawPath = String(componentData.fileData.filePath).replaceAll("\\", "/")

                // 保证最终路径是 GitHub repo 里的 content/xxx.md
                const contentIndex = rawPath.lastIndexOf("/content/")
                const repoPath =
                  contentIndex >= 0
                    ? rawPath.slice(contentIndex + 1)
                    : rawPath.startsWith("content/")
                      ? rawPath
                      : `content/${rawPath}`

                const encodedPath = repoPath.split("/").map(encodeURIComponent).join("/")
                const editUrl = `https://github.com/yifu-ding/notes/edit/v5/${encodedPath}`

                return (
                  <div class="edit-on-github">
                    <a href={editUrl} target="_blank" rel="noopener noreferrer">
                      Edit this page on GitHub
                    </a>
                  </div>
                )
              })()}
            <Header {...componentData}>
              {header.map((HeaderComponent) => (
                <HeaderComponent {...componentData} />
              ))}
            </Header>
            <div class="popover-hint">
              {beforeBody.map((BodyComponent) => (
                <BodyComponent {...componentData} />
              ))}
            </div>
          </div>
          <Content {...componentData} />
          <hr />
          <div class="page-footer">
            {afterBody.length > 0 &&
              componentData.fileData.frontmatter?.comments !== false &&
              componentData.fileData.frontmatter?.comments !== "false" && (
                <div class="comments-notice">
                  <h2>讨论</h2>
                  <p>
                    评论区由 GitHub Discussions 提供支持。发表评论需要使用 GitHub 账号登录，您发布的内容将公开显示在{" "}
                    <a
                      href="https://github.com/yifu-ding/notes"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      本仓库
                    </a>{" "}
                    的 Discussions 页面中。如需删除已发表的评论，请前往{" "}
                    <a
                      href="https://github.com/yifu-ding/notes/discussions"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Discussions 页面
                    </a>{" "}
                    找到对应评论并自行删除。
                  </p>
                </div>
              )}
            {afterBody.map((BodyComponent) => (
              <BodyComponent {...componentData} />
            ))}
          </div>
        </div>
        <div class="right sidebar">
          {right.map((BodyComponent) => (
            <BodyComponent {...componentData} />
          ))}
        </div>
        {footer.map((FooterComponent) => (
          <FooterComponent {...componentData} />
        ))}
      </>
    )
  },
}
