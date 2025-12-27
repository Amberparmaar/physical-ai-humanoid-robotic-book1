import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import FloatingChatbot from '@site/src/components/FloatingChatbot';

export default function LayoutWrapper(props) {
  return (
    <>
      <OriginalLayout {...props}>
        {props.children}
      </OriginalLayout>
      <FloatingChatbot />
    </>
  );
}