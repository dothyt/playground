#ifndef _REMOTE_TAGGING_H_
#define _REMOTE_TAGGING_H_

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include <tuple>

#define SMALLSET_SIZE 20
#define SMALLVECTOR_SIZE 400

struct TypeTracker : public llvm::PassInfoMixin<TypeTracker> {
	llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &) {
		runModule(M);
		return llvm::PreservedAnalyses::all();
	}
													
	void runModule(llvm::Module &M);
	inline bool ifNeedTag(llvm::StructType *type) {
		return types_to_tag.find(type) != types_to_tag.end();
	}
	inline bool typeVisited(llvm::StructType *type) {
		return visited_types.find(type) 
				!= visited_types.end();
	}
	bool containsRemoteType(llvm::Type *type);
	bool structContainsRemoteType(llvm::StructType *type);
	bool ptrPointToRemoteType(llvm::Type *type);
	// void constructTypeMembersToTag(llvm::Module &M);

	static bool isRequired() { return true; }

	llvm::SmallSet<llvm::StructType *, SMALLSET_SIZE> types_to_tag;
	llvm::SmallSet<llvm::StructType *, SMALLSET_SIZE> visited_types;
	llvm::SmallVector<
		std::tuple<llvm::StructType *, llvm::Type::TypeID, unsigned, int>, 
		SMALLVECTOR_SIZE
	> type_members_to_tag;
};

struct InjectFuncCall : public llvm::PassInfoMixin<InjectFuncCall> {
	llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
		return runModule(M, AM) ? llvm::PreservedAnalyses::none()
							: llvm::PreservedAnalyses::all();
	}
	bool runModule(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
	static bool isRequired() { return true; }
};

#endif /* _REMOTE_TAGGING_H_ */