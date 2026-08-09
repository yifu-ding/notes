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
          <hr />
          <div class="page-footer">
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
